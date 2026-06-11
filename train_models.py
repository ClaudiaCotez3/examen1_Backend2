"""
Entrenamiento OFFLINE del motor de enrutamiento y riesgos (TensorFlow).

Lee el historial de trámites desde Mongo, entrena tres modelos Keras y
los persiste en MODELS_DIR para que `routing_engine.py` solo tenga que
cargarlos e inferir:

  * eta.keras     — regresión de la demora (log1p de los minutos).
  * risk.keras    — clasificación binaria del riesgo de atraso.
  * anomaly.keras — autoencoder para detección de anomalías.
  * encoders.json — vocabularios + escaladores + umbrales.

Uso:
    python train_models.py                # entrena con datos reales de Mongo
    python train_models.py --synthetic 800  # genera 800 muestras sintéticas
    python train_models.py --epochs 60

El modo --synthetic existe para poder entrenar/demostrar el motor aunque
la base aún tenga poco historial (caso típico en un parcial). Genera un
dataset realista en memoria; NO escribe nada en Mongo.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

from routing_engine import (
    CATEGORICAL_FIELDS,
    Encoders,
    UNKNOWN,
    _models_dir,
    _time_features,
    build_training_frame,
    featurize,
    ANOMALY_MODEL_PATH,
    ETA_MODEL_PATH,
    RISK_MODEL_PATH,
)


# ── Datos sintéticos (para demo / poco historial) ─────────────────────


def build_synthetic_frame(n: int, seed: int = 42) -> list[dict]:
    """Genera `n` instancias plausibles. La demora depende de la
    actividad, de la eficiencia del operador y del backlog, más ruido —
    justo el tipo de relación no lineal que la red debe aprender."""
    rng = np.random.default_rng(seed)
    activities = [f"act_{i}" for i in range(6)]
    lanes = [f"lane_{i}" for i in range(3)]
    operators = [f"op_{i}" for i in range(5)]
    # base de minutos por actividad y "factor" de eficiencia por operador.
    act_base = {a: float(rng.uniform(15, 120)) for a in activities}
    op_factor = {o: float(rng.uniform(0.6, 1.6)) for o in operators}
    act_lane = {a: lanes[i % len(lanes)] for i, a in enumerate(activities)}

    rows = []
    base_day = 1_700_000_000  # epoch fijo (Date.now no está disponible aquí)
    for i in range(n):
        a = activities[int(rng.integers(0, len(activities)))]
        o = operators[int(rng.integers(0, len(operators)))]
        backlog = float(rng.integers(0, 12))
        from datetime import datetime, timezone
        created = datetime.fromtimestamp(base_day + i * 137, tz=timezone.utc)
        noise = float(rng.normal(0, 8))
        service = max(1.0, act_base[a] * op_factor[o] * (1 + 0.08 * backlog) + noise)
        wait = max(0.0, backlog * float(rng.uniform(2, 6)) + float(rng.normal(0, 4)))
        rows.append({
            "instance_id": f"syn_{i}",
            "activity_id": a,
            "operator_id": o,
            "lane_id": act_lane[a],
            "created": created,
            "backlog": backlog,
            "lead": service + wait,
            "wait": wait,
            "service": service,
        })
    return rows


# ── Encoders: vocabularios, escaladores, umbrales ─────────────────────


def fit_encoders(frame: list[dict]) -> Encoders:
    enc = Encoders()

    # Vocabularios categóricos (idx 0 reservado a desconocido).
    field_to_key = {"activity": "activity_id", "operator": "operator_id", "lane": "lane_id"}
    for field in CATEGORICAL_FIELDS:
        key = field_to_key[field]
        values = sorted({str(r[key]) for r in frame if r.get(key) and r[key] != UNKNOWN})
        enc.vocabs[field] = {v: i + 1 for i, v in enumerate(values)}

    # Escalador numérico (hour_sin, hour_cos, dow_sin, dow_cos, backlog).
    raw = []
    for r in frame:
        hs, hc, ds, dc = _time_features(r.get("created"))
        raw.append([hs, hc, ds, dc, float(r.get("backlog", 0.0))])
    raw_arr = np.array(raw, dtype=np.float64)
    enc.numeric_mean = raw_arr.mean(axis=0).tolist()
    enc.numeric_std = (raw_arr.std(axis=0) + 1e-6).tolist()

    # Umbral de riesgo por actividad = mediana del lead × 1.5.
    by_act: dict[str, list[float]] = {}
    for r in frame:
        by_act.setdefault(str(r["activity_id"]), []).append(float(r["lead"]))
    enc.risk_threshold = {a: float(np.median(v) * 1.5) for a, v in by_act.items()}

    # Escalador del autoencoder (lead, wait, service).
    anomaly_raw = np.array([[r["lead"], r["wait"], r["service"]] for r in frame], dtype=np.float64)
    enc.anomaly_mean = anomaly_raw.mean(axis=0).tolist()
    enc.anomaly_std = (anomaly_raw.std(axis=0) + 1e-6).tolist()
    return enc


def risk_labels(frame: list[dict], enc: Encoders) -> np.ndarray:
    out = []
    for r in frame:
        thr = enc.risk_threshold.get(str(r["activity_id"]), float("inf"))
        out.append(1.0 if float(r["lead"]) > thr else 0.0)
    return np.array(out, dtype=np.float32)


# ── Arquitecturas Keras ───────────────────────────────────────────────


def _shared_trunk(keras, layers, enc: Encoders):
    """Tronco compartido: tres embeddings (actividad/operador/calle) + la
    rama numérica, concatenados."""
    inputs = {}
    parts = []
    for field in CATEGORICAL_FIELDS:
        size = enc.vocab_size(field)
        dim = max(2, min(8, size // 2))
        inp = keras.Input(shape=(), dtype="int32", name=field)
        emb = layers.Embedding(input_dim=size, output_dim=dim)(inp)
        parts.append(layers.Flatten()(emb))
        inputs[field] = inp
    num_in = keras.Input(shape=(5,), dtype="float32", name="numeric")
    inputs["numeric"] = num_in
    parts.append(num_in)
    x = layers.Concatenate()(parts)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    return inputs, x


def build_eta_model(keras, layers, enc: Encoders):
    inputs, x = _shared_trunk(keras, layers, enc)
    out = layers.Dense(1, activation="linear", name="eta")(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_risk_model(keras, layers, enc: Encoders):
    inputs, x = _shared_trunk(keras, layers, enc)
    out = layers.Dense(1, activation="sigmoid", name="risk")(x)
    model = keras.Model(inputs=inputs, outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_autoencoder(keras, layers):
    inp = keras.Input(shape=(3,), dtype="float32", name="anomaly_in")
    e = layers.Dense(8, activation="relu")(inp)
    e = layers.Dense(2, activation="relu")(e)
    d = layers.Dense(8, activation="relu")(e)
    out = layers.Dense(3, activation="linear")(d)
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Orquestación ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el motor TF de enrutamiento y riesgos.")
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Genera N muestras sintéticas en vez de leer Mongo.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if args.synthetic > 0:
        print(f"[*] Generando {args.synthetic} muestras sintéticas…")
        frame = build_synthetic_frame(args.synthetic)
    else:
        print("[*] Leyendo historial real desde Mongo…")
        frame = build_training_frame()

    if len(frame) < 20:
        print(f"[!] Solo hay {len(frame)} muestras. Es muy poco para entrenar; "
              f"usa --synthetic 800 para una demo, o acumula más historial.")
        if len(frame) == 0:
            return

    print(f"[ok] {len(frame)} muestras. Ajustando encoders…")
    enc = fit_encoders(frame)
    for field in CATEGORICAL_FIELDS:
        print(f"    vocab[{field}] = {len(enc.vocabs[field])}")

    # Import perezoso: solo aquí necesitamos TensorFlow.
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = featurize(frame, enc)
    eta_target = np.log1p(np.array([r["lead"] for r in frame], dtype=np.float32))
    risk_target = risk_labels(frame, enc)
    print(f"    riesgo: {int(risk_target.sum())}/{len(risk_target)} muestras positivas")

    os.makedirs(_models_dir(), exist_ok=True)

    print("[*] Entrenando ETA (demora)…")
    eta_model = build_eta_model(keras, layers, enc)
    eta_model.fit(inputs, eta_target, epochs=args.epochs, batch_size=args.batch,
                  validation_split=0.15, verbose=2)
    eta_model.save(ETA_MODEL_PATH())

    print("[*] Entrenando RIESGO…")
    risk_model = build_risk_model(keras, layers, enc)
    risk_model.fit(inputs, risk_target, epochs=args.epochs, batch_size=args.batch,
                   validation_split=0.15, verbose=2)
    risk_model.save(RISK_MODEL_PATH())

    print("[*] Entrenando ANOMALÍAS (autoencoder)…")
    anomaly_X = np.array([enc.scale_anomaly([r["lead"], r["wait"], r["service"]]) for r in frame],
                         dtype=np.float32)
    ae = build_autoencoder(keras, layers)
    ae.fit(anomaly_X, anomaly_X, epochs=args.epochs, batch_size=args.batch,
           validation_split=0.15, verbose=2)
    ae.save(ANOMALY_MODEL_PATH())

    # Umbral de anomalía = media + 2σ del error de reconstrucción.
    recon = ae.predict(anomaly_X, verbose=0)
    errors = np.mean(np.square(anomaly_X - recon), axis=1)
    enc.anomaly_threshold = float(errors.mean() + 2 * errors.std())
    print(f"    umbral anomalía = {enc.anomaly_threshold:.4f}")

    enc.save()
    print(f"[ok] Listo. Modelos + encoders guardados en {_models_dir()}")


if __name__ == "__main__":
    main()
