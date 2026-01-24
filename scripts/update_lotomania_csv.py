#!/usr/bin/env python3
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime
from typing import List, Dict, Any, Optional

API_BASE = os.getenv("LOTOMANIA_API_BASE", "https://loteriascaixa-api.herokuapp.com/api")
CSV_PATH = "lotomania_historico_onehot.csv"

# Formato: concurso;data;00;01;...;99  (100 colunas binárias)
HEADER = ["concurso", "data"] + [f"{i:02d}" for i in range(100)]


def http_get_json(url: str, timeout: int = 30) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "github-actions-lotomania-bot"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def parse_latest() -> Dict[str, Any]:
    # endpoint: /api/<loteria>/latest
    url = f"{API_BASE}/lotomania/latest"
    return http_get_json(url)


def parse_by_concurso(concurso: int) -> Optional[Dict[str, Any]]:
    # muitas instalações expõem /api/<loteria>/<concurso>
    url = f"{API_BASE}/lotomania/{concurso}"
    try:
        return http_get_json(url)
    except Exception:
        return None


def dezenas_to_onehot(dezenas: List[str]) -> List[int]:
    s = set(int(d) for d in dezenas)
    out = []
    for i in range(100):
        out.append(1 if i in s else 0)
    # checagem: Lotomania deve ter 20 dezenas
    if sum(out) != 20:
        raise ValueError(f"Esperava 20 dezenas, veio {sum(out)}: {sorted(s)}")
    return out


def normalize_date(d: str) -> str:
    # API costuma vir "DD/MM/YYYY"; seu CSV usa o mesmo padrão
    try:
        datetime.strptime(d, "%d/%m/%Y")
        return d
    except Exception:
        return d


def ensure_csv_exists():
    if os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(HEADER)


def read_existing_concursos() -> Dict[int, str]:
    out: Dict[int, str] = {}
    if not os.path.exists(CSV_PATH):
        return out
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f, delimiter=";")
        header = next(r, None)
        if not header:
            return out
        for row in r:
            if not row or len(row) < 2:
                continue
            try:
                c = int(row[0])
                out[c] = row[1]
            except Exception:
                continue
    return out


def append_row(concurso: int, data: str, dezenas: List[str]):
    onehot = dezenas_to_onehot(dezenas)
    row = [str(concurso), normalize_date(data)] + [str(x) for x in onehot]
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(row)


def main():
    ensure_csv_exists()
    existing = read_existing_concursos()

    latest = parse_latest()
    latest_concurso = int(latest["concurso"])
    latest_data = latest.get("data", "")
    latest_dezenas = latest.get("dezenas") or latest.get("dezenasOrdemSorteio")

    if not latest_dezenas:
        raise RuntimeError("API não retornou dezenas no /latest.")

    # Se já existe, nada a fazer
    if latest_concurso in existing:
        print(f"CSV já está no concurso {latest_concurso}. Nada a fazer.")
        return 0

    # Tenta preencher a lacuna entre o maior concurso do CSV e o latest
    max_existing = max(existing.keys()) if existing else None
    start = (max_existing + 1) if max_existing is not None else latest_concurso

    print(f"Maior concurso no CSV: {max_existing}. Latest na API: {latest_concurso}.")
    print(f"Vou tentar preencher de {start} até {latest_concurso}.")

    for c in range(start, latest_concurso + 1):
        if c in existing:
            continue

        if c == latest_concurso:
            append_row(c, latest_data, latest_dezenas)
            print(f"Adicionado concurso {c} (latest).")
            continue

        payload = parse_by_concurso(c)
        if payload and ("dezenas" in payload or "dezenasOrdemSorteio" in payload):
            dezenas = payload.get("dezenas") or payload.get("dezenasOrdemSorteio")
            data = payload.get("data", "")
            append_row(c, data, dezenas)
            print(f"Adicionado concurso {c}.")
        else:
            print(f"Não consegui obter concurso {c} via /lotomania/{c}. Pulando.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
