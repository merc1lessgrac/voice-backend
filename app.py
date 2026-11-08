import os
import time
import joblib
import requests
import librosa
import opensmile
import parselmouth
from parselmouth.praat import call
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

BOT_TOKEN = os.getenv("BOT_TOKEN", "9096266573:AAEY199yjgfwIVBC8mCrE6c92LLcC9DlwbI")
CHAT_ID = os.getenv("CHAT_ID", "1105879494")

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


bundle = joblib.load("depression_voice_model.pkl")
ensemble = bundle['model']
sex_stats = bundle['sex_stats']
best_thresh = bundle['threshold']
GOLDEN_FEATURES = bundle['features']

smile_engine = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

@app.get("/", response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    sex: str = Form(...),
    age: int = Form(...),
    voice_clean: str = Form(...),
    smoker: str = Form(...),
    mood: int = Form(...)
):
    session_id = int(time.time()) % 1000000
    ext = os.path.splitext(audio.filename)[1] or ".webm"
    temp_audio = f"temp_{session_id}{ext}"
    with open(temp_audio, "wb") as f:
        f.write(await audio.read())

    try:
        y, sr = librosa.load(temp_audio, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        total_duration = len(y_trimmed) / sr
        if total_duration < 8.0:
            return JSONResponse({"error": "Запись слишком короткая (менее 8 секунд). Пожалуйста, прочитайте текст полностью."})

        sex_id = 0 if sex == "Мужской" else 1
        snd = parselmouth.Sound(y_trimmed, sampling_frequency=sr)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=400.0)
        p_vals = pitch.selected_array['frequency']
        p_vals = p_vals[p_vals > 0]
        f0_median = float(np.median(p_vals)) if len(p_vals) > 0 else 150.0
        f0_cv = float(np.std(p_vals) / (np.mean(p_vals) + 1e-6)) if len(p_vals) > 0 else 0.0

        pp = call(snd, "To PointProcess (periodic, cc)", 65.0, 400.0)
        jitter_local = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harm = call(snd, "To Harmonicity (cc)", 0.01, 65.0, 0.1, 1.0)
        hnr_mean = call(harm, "Get mean", 0, 0)

        non_mute = librosa.effects.split(y_trimmed, top_db=28)
        speech_time = sum([(end - start) / sr for start, end in non_mute])
        speech_ratio = speech_time / total_duration
        pauses = [(non_mute[i][0] - non_mute[i-1][1]) / sr for i in range(1, len(non_mute))]
        pauses = [p for p in pauses if p >= 0.15]
        pause_count = len(pauses)
        pause_mean = float(np.mean(pauses)) if pause_count > 0 else 0.0

        egemaps_dict = smile_engine.process_signal(y_trimmed, sr).iloc[0].to_dict()
        raw_golden = {
            'total_duration_sec': total_duration,
            'pause_mean_sec': pause_mean,
            'spectralFlux_sma3_stddevNorm': float(egemaps_dict['spectralFlux_sma3_stddevNorm']),
            'F2amplitudeLogRelF0_sma3nz_stddevNorm': float(egemaps_dict['F2amplitudeLogRelF0_sma3nz_stddevNorm']),
            'mfcc3V_sma3nz_stddevNorm': float(egemaps_dict['mfcc3V_sma3nz_stddevNorm']),
            'loudnessPeaksPerSec': float(egemaps_dict['loudnessPeaksPerSec'])
        }

        z_dict, z_vector = {}, []
        for col in GOLDEN_FEATURES:
            m = sex_stats[sex_id][col]['mean']
            s = sex_stats[sex_id][col]['std']
            z_val = (raw_golden[col] - m) / s
            z_dict[col] = z_val
            z_vector.append(z_val)

        X_new = pd.DataFrame([z_vector], columns=GOLDEN_FEATURES)
        prob = float(ensemble.predict_proba(X_new)[0, 1])
        is_depressed = prob >= best_thresh

        # Полная таблица для отправки исследователю в Telegram
        row = {
            'session_id': session_id, 'sex': sex, 'age': age, 'smoker': smoker,
            'voice_clean': voice_clean, 'self_mood': mood, 'AI_prob': round(prob, 4),
            'AI_verdict': int(is_depressed), 'f0_hz': round(f0_median, 2),
            'duration_sec': round(total_duration, 2), 'speech_ratio': round(speech_ratio, 4),
            'pause_count': pause_count, 'pause_mean_sec': round(pause_mean, 3),
            'jitter': round(jitter_local, 5), 'shimmer': round(shimmer_local, 5), 'hnr': round(hnr_mean, 3)
        }
        row.update(egemaps_dict)
        excel_path = f"report_{session_id}.xlsx"
        pd.DataFrame([row]).to_excel(excel_path, index=False)

        tg_text = (
            f"Новое исследование #{session_id}\n"
            f"Пол: {sex} | Возраст: {age} | Самооценка: {mood}/10\n"
            f"Чистый голос: {voice_clean} | Курение: {smoker}\n"
            f"Длительность: {total_duration:.1f} сек (Z={z_dict['total_duration_sec']:+.2f})\n"
            f"Средняя пауза: {pause_mean:.2f} сек (Пауз: {pause_count})\n"
            f"Вероятность модели: {prob*100:.1f}% (Порог: {best_thresh*100:.1f}%)\n"
            f"Заключение: {'ВЫЯВЛЕН РИСК' if is_depressed else 'НОРМА'}"
        )
        try:
            requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": tg_text})
            with open(temp_audio, "rb") as af:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument", data={"chat_id": CHAT_ID}, files={"document": af})
            with open(excel_path, "rb") as ef:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument", data={"chat_id": CHAT_ID}, files={"document": ef})
        except Exception:
            pass

        if os.path.exists(excel_path): os.remove(excel_path)
        if os.path.exists(temp_audio): os.remove(temp_audio)

        return JSONResponse({
            "session_id": session_id,
            "is_depressed": is_depressed,
            "status_title": "Выявлены маркеры психомоторного утомления и снижения речевого тонуса" if is_depressed else "Акустические и психомоторные показатели в пределах физиологической нормы",
            "duration": round(total_duration, 1),
            "norm_duration": round(sex_stats[sex_id]['total_duration_sec']['norm_median'], 0),
            "tempo_status": "Снижен" if z_dict['total_duration_sec'] > 0.45 else "В пределах нормы",
            "speech_ratio": round(speech_ratio * 100, 1),
            "pause_mean": round(pause_mean, 2),
            "pause_status": "Удлинены паузы" if z_dict['pause_mean_sec'] > 0.45 else "Ритмичное, без задержек",
            "f0": round(f0_median, 1),
            "peaks": round(raw_golden['loudnessPeaksPerSec'], 1),
            "artic_status": "Снижение тонуса артикуляции" if z_dict['F2amplitudeLogRelF0_sma3nz_stddevNorm'] > 0.45 else "Четкая, выразительная"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
