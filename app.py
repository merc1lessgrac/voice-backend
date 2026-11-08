import os
import time
import subprocess
import joblib
import requests
import librosa
import soundfile as sf
import opensmile
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

app = FastAPI(title="Voice Biomarkers Screening API")

# Полное разрешение кросс-доменных запросов (CORS) для фронтенда на Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка обученного диагностического ядра
bundle = joblib.load("depression_voice_model.pkl")
ensemble = bundle['model']
sex_stats = bundle['sex_stats']
best_thresh = bundle['threshold']
GOLDEN_FEATURES = bundle['features']

# Инициализация акустического экстрактора eGeMAPSv02
smile_engine = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def send_to_telegram_background(session_id: int, audio_path: str, excel_path: str, tg_text: str):
    """Асинхронная выгрузка протокола в Telegram без задержки ответа клиенту."""
    try:
        if BOT_TOKEN and CHAT_ID:
            requests.post(
                f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                data={"chat_id": CHAT_ID, "text": tg_text},
                timeout=25
            )
            with open(audio_path, "rb") as af:
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                    data={"chat_id": CHAT_ID},
                    files={"document": af},
                    timeout=30
                )
            with open(excel_path, "rb") as ef:
                requests.post(
                    f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument",
                    data={"chat_id": CHAT_ID},
                    files={"document": ef},
                    timeout=25
                )
    except Exception as exc:
        print(f"[TG_ERROR] Ошибка при отправке сессии #{session_id}: {exc}")
    finally:
        if os.path.exists(excel_path):
            os.remove(excel_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)

@app.get("/")
def healthcheck():
    """Эндпоинт для проверки статуса и моментального пробуждения сервиса."""
    return {"status": "ready", "service": "Voice Biomarkers Screening API"}

@app.post("/api/analyze")
async def analyze_audio(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    sex: str = Form(...),
    age: int = Form(...),
    voice_clean: str = Form(...),
    smoker: str = Form(...),
    mood: int = Form(...)
):
    session_id = int(time.time()) % 1000000
    ext = os.path.splitext(audio.filename)[1] or ".tmp"
    raw_audio_path = f"raw_{session_id}{ext}"
    clean_wav_path = f"clean_{session_id}.wav"

    # Сохранение входящего потока
    with open(raw_audio_path, "wb") as f:
        f.write(await audio.read())

    try:
        # Быстрая нормализация в 16 кГц моно через системный ffmpeg (исключает зависания audioread)
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_audio_path, "-ar", "16000", "-ac", "1", clean_wav_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        if os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)

        # Прямое чтение подготовленного PCM WAV
        y, sr = sf.read(clean_wav_path)
        y = y.astype(np.float32)

        # Тримминг тишины в начале и конце
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        total_duration = len(y_trimmed) / sr

        if total_duration < 8.0:
            if os.path.exists(clean_wav_path):
                os.remove(clean_wav_path)
            return JSONResponse({"error": "Длительность звучания менее 8 секунд. Пожалуйста, прочитайте текст полностью."})

        sex_id = 0 if sex == "Мужской" else 1

        # Спектрально-формантный анализ через Praat
        snd = parselmouth.Sound(y_trimmed, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=400.0)
        p_vals = pitch.selected_array['frequency']
        p_vals = p_vals[p_vals > 0]
        f0_median = float(np.median(p_vals)) if len(p_vals) > 0 else 150.0

        pp = call(snd, "To PointProcess (periodic, cc)", 65.0, 400.0)
        jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 65.0, 0.1, 1.0)
        hnr_mean = call(harm, "Get mean", 0, 0)

        # Анализ психомоторных пауз и коэффициента фонации
        non_mute = librosa.effects.split(y_trimmed, top_db=28)
        speech_time = sum([(end - start) / sr for start, end in non_mute])
        speech_ratio = speech_time / total_duration
        pauses = [(non_mute[i][0] - non_mute[i - 1][1]) / sr for i in range(1, len(non_mute))]
        pauses = [p for p in pauses if p >= 0.15]
        pause_count = len(pauses)
        pause_mean = float(np.mean(pauses)) if pause_count > 0 else 0.0

        # Экстракция стандарта eGeMAPSv02
        egemaps_dict = smile_engine.process_signal(y_trimmed, sr).iloc[0].to_dict()

        raw_golden = {
            'total_duration_sec': total_duration,
            'pause_mean_sec': pause_mean,
            'spectralFlux_sma3_stddevNorm': float(egemaps_dict['spectralFlux_sma3_stddevNorm']),
            'F2amplitudeLogRelF0_sma3nz_stddevNorm': float(egemaps_dict['F2amplitudeLogRelF0_sma3nz_stddevNorm']),
            'mfcc3V_sma3nz_stddevNorm': float(egemaps_dict['mfcc3V_sma3nz_stddevNorm']),
            'loudnessPeaksPerSec': float(egemaps_dict['loudnessPeaksPerSec'])
        }

        # Гендерная Z-нормализация признаков
        z_dict = {}
        z_vector = []
        for col in GOLDEN_FEATURES:
            m = sex_stats[sex_id][col]['mean']
            s = sex_stats[sex_id][col]['std']
            z_val = (raw_golden[col] - m) / s
            z_dict[col] = z_val
            z_vector.append(z_val)

        # Предсказание ансамбля
        x_input = pd.DataFrame([z_vector], columns=GOLDEN_FEATURES)
        prob = float(ensemble.predict_proba(x_input)[0, 1])
        is_depressed = prob >= best_thresh

        # Генерация полной исследовательской таблицы Excel
        row_data = {
            'session_id': session_id, 'sex': sex, 'age': age, 'smoker': smoker,
            'voice_clean': voice_clean, 'self_mood': mood, 'AI_prob': round(prob, 4),
            'AI_verdict': int(is_depressed), 'f0_hz': round(f0_median, 2),
            'duration_sec': round(total_duration, 2), 'speech_ratio': round(speech_ratio, 4),
            'pause_count': pause_count, 'pause_mean_sec': round(pause_mean, 3),
            'jitter': round(jitter_local, 5), 'shimmer': round(shimmer_local, 5), 'hnr': round(hnr_mean, 3)
        }
        row_data.update(egemaps_dict)
        excel_path = f"protocol_{session_id}.xlsx"
        pd.DataFrame([row_data]).to_excel(excel_path, index=False)

        tg_caption = (
            f"Исследование #{session_id}\n"
            f"Пол: {sex} | Возраст: {age} | Самооценка: {mood}/10\n"
            f"Чистый голос: {voice_clean} | Курение: {smoker}\n"
            f"Длительность: {total_duration:.1f} с (Z={z_dict['total_duration_sec']:+.2f})\n"
            f"Паузы: {pause_mean:.2f} с (Кол-во: {pause_count})\n"
            f"Вероятность депрессии: {prob*100:.1f}% (Порог: {best_thresh*100:.1f}%)\n"
            f"Статус: {'РИСК ДЕПРЕССИИ' if is_depressed else 'НОРМА'}"
        )

        # Выгрузка в Telegram в фоновом режиме
        background_tasks.add_task(send_to_telegram_background, session_id, clean_wav_path, excel_path, tg_caption)

        # Мгновенный возврат клинического заключения в браузер
        return JSONResponse({
            "session_id": session_id,
            "is_depressed": is_depressed,
            "status_title": "Выявлены маркеры психомоторного утомления и снижения речевого тонуса" if is_depressed else "Показатели речевой моторики в пределах возрастной нормы",
            "duration": round(total_duration, 1),
            "norm_duration": round(sex_stats[sex_id]['total_duration_sec']['norm_median'], 0),
            "tempo_status": "Замедлен" if z_dict['total_duration_sec'] > 0.45 else "В пределах нормы",
            "speech_ratio": round(speech_ratio * 100, 1),
            "pause_mean": round(pause_mean, 2),
            "pause_status": "Удлинены паузы" if z_dict['pause_mean_sec'] > 0.45 else "Ритмичное дыхание",
            "f0": round(f0_median, 1),
            "peaks": round(raw_golden['loudnessPeaksPerSec'], 1),
            "artic_status": "Снижение вариативности" if z_dict['F2amplitudeLogRelF0_sma3nz_stddevNorm'] > 0.45 else "Выразительная"
        })

    except Exception as exc:
        if os.path.exists(raw_audio_path):
            os.remove(raw_audio_path)
        if os.path.exists(clean_wav_path):
            os.remove(clean_wav_path)
        return JSONResponse({"error": f"Сбой обработки биомаркеров: {str(exc)}"}, status_code=500)
