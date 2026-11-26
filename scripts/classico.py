#!/usr/bin/env python3
"""Pipeline clássica de contagem de parafusos, porcas e arruelas (Fase 3).

O que faz: processa cada imagem em 7 etapas, todas parametrizadas por um dicionário de
configuração: etapa1_preprocessar, etapa2_segmentar, etapa3_morfologia,
etapa4_separar_watershed, etapa5_componentes, etapa6_classificar_componentes e
etapa7_contar_por_classe. A calibração é feita por dataset, usando só train e valid.
O test só é lido no modo avaliar-test, e uma única vez.

Entradas: data_final/base/<dataset>/<divisão>/{images,labels} e data_final/contagens_gt.csv.
Saídas, em resultados/classico/:
  - config_final.json e config_final.sha256 (configuração congelada)
  - previsoes_valid.csv e previsoes_test.csv (formato lido por scripts/avaliar.py)
  - total_por_imagem_valid.csv e total_por_imagem.csv (diagnóstico do total de objetos)
  - descritores_train.csv, histogramas/ e figuras/

Como rodar:
  # 1) Calibra em train/valid, gera os histogramas e congela config_final.json
  .venv/bin/python scripts/classico.py --modo calibrar

  # 2) Reaplica a configuração congelada no valid
  .venv/bin/python scripts/classico.py --modo avaliar-valid

  # 3) Avaliação final no test (uma única vez), conferindo o SHA-256 de config_final.json
  .venv/bin/python scripts/classico.py --modo avaliar-test --hash-esperado <SHA256>
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_BASE = RAIZ_PROJETO / "data_final" / "base"
ARQUIVO_GT = RAIZ_PROJETO / "data_final" / "contagens_gt.csv"
ARQUIVO_EXCLUSOES_TEST = RAIZ_PROJETO / "resultados" / "auditoria" / "exclusoes_teste.csv"
PASTA_SAIDA_CLASSICO = RAIZ_PROJETO / "resultados" / "classico"
PASTA_FIGURAS = PASTA_SAIDA_CLASSICO / "figuras"
PASTA_HISTOGRAMAS = PASTA_SAIDA_CLASSICO / "histogramas"
ARQUIVO_CONFIG_FINAL = PASTA_SAIDA_CLASSICO / "config_final.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avaliar as av  # noqa: E402

DATASETS_CLASSES: Dict[str, Tuple[str, ...]] = {
    "washers_hardee": ("arruela",),
    "roboflow_screws": ("parafuso",),
    "nutsnbolts": ("parafuso", "porca", "arruela"),
    "screws_and_nuts": ("parafuso", "porca"),
}

ID_PARA_CLASSE = {0: "parafuso", 1: "porca", 2: "arruela"}
CLASSE_PARA_ID = {"parafuso": 0, "porca": 1, "arruela": 2}
CORES_BGR = {
    "parafuso": (0, 140, 255),
    "porca": (255, 90, 0),
    "arruela": (0, 190, 0),
}


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def sha256_arquivo(caminho: Path) -> str:
    """SHA-256 do arquivo inteiro (usado para congelar config_final.json)."""
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _impar(k: int) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def detectar_padding_rotacao(cinza: np.ndarray, limiar: int) -> np.ndarray:
    """Marca as regiões pretas ligadas à borda da imagem (preenchimento deixado pela rotação do Roboflow).

    Usa floodFill a partir dos pixels de borda com valor <= limiar.
    Devolve máscara booleana (True = pixel de padding, a ser ignorado).
    """
    h, w = cinza.shape
    mask_preto = (cinza <= limiar).astype(np.uint8) * 255
    flood = np.zeros((h + 2, w + 2), np.uint8)
    work = mask_preto.copy()

    # Semeia o preenchimento em todo pixel de borda com valor <= limiar.
    for x in range(w):
        if work[0, x] == 255:
            cv2.floodFill(work, flood, (x, 0), 128)
        if work[h - 1, x] == 255:
            cv2.floodFill(work, flood, (x, h - 1), 128)
    for y in range(h):
        if work[y, 0] == 255:
            cv2.floodFill(work, flood, (0, y), 128)
        if work[y, w - 1] == 255:
            cv2.floodFill(work, flood, (w - 1, y), 128)

    padding = work == 128
    # Se a região cobrir mais de 55% da imagem, não é padding de rotação: descarta a detecção.
    frac = float(padding.mean())
    if frac > 0.55:
        return np.zeros((h, w), dtype=bool)
    return padding


# =============================================================================
# ETAPA 1 — PRÉ-PROCESSAMENTO
# =============================================================================

def etapa1_preprocessar(bgr: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pré-processamento configurável:
      - conversão para escala de cinza
      - filtro de mediana
      - correção de iluminação opcional (gauss_divisao, morf_abertura, tophat, blackhat)
      - CLAHE opcional
    Retorna (imagem_preprocessada_uint8, mascara_padding_bool).
    """
    cinza = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    padding_limiar = int(cfg["padding_limiar"])
    mascara_padding = (
        detectar_padding_rotacao(cinza, padding_limiar)
        if bool(cfg["ignorar_padding_preto"])
        else np.zeros(cinza.shape, dtype=bool)
    )

    k_med = _impar(int(cfg["mediana_k"]))
    proc = cv2.medianBlur(cinza, k_med) if k_med >= 3 else cinza.copy()

    metodo_ilum = str(cfg["iluminacao_metodo"])
    if metodo_ilum == "gauss_divisao":
        sigma = float(cfg["iluminacao_sigma"])
        intensidade = float(cfg["iluminacao_intensidade"])
        g_min = float(cfg["iluminacao_ganho_min"])
        g_max = float(cfg["iluminacao_ganho_max"])
        orig_f = proc.astype(np.float32)
        fundo = cv2.GaussianBlur(orig_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        ref = float(np.median(fundo[~mascara_padding])) if (~mascara_padding).any() else float(np.median(fundo))
        ganho = np.clip(ref / np.maximum(fundo, 1.0), g_min, g_max)
        corrigida = orig_f * (1.0 + intensidade * (ganho - 1.0))
        dif = float(np.mean(orig_f) - np.mean(corrigida))
        proc = np.rint(np.clip(corrigida + dif, 0, 255)).astype(np.uint8)
    elif metodo_ilum == "morf_abertura":
        k_il = _impar(int(cfg["iluminacao_k"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_il, k_il))
        fundo = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel).astype(np.float32)
        orig_f = proc.astype(np.float32)
        ref = float(np.median(fundo))
        ganho = np.clip(ref / np.maximum(fundo, 1.0), 0.7, 1.4)
        proc = np.rint(np.clip(orig_f * ganho, 0, 255)).astype(np.uint8)
    elif metodo_ilum == "tophat":
        k_il = _impar(int(cfg["iluminacao_k"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_il, k_il))
        proc = cv2.morphologyEx(proc, cv2.MORPH_TOPHAT, kernel)
    elif metodo_ilum == "blackhat":
        k_il = _impar(int(cfg["iluminacao_k"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_il, k_il))
        proc = cv2.morphologyEx(proc, cv2.MORPH_BLACKHAT, kernel)
    elif metodo_ilum != "nenhum":
        raise ValueError(f"iluminacao_metodo desconhecido: {metodo_ilum}")

    if bool(cfg["usar_clahe"]):
        clip = float(cfg["clahe_clip"])
        grid = int(cfg["clahe_grid"])
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        proc = clahe.apply(proc)

    return proc, mascara_padding


# =============================================================================
# ETAPA 2 — SEGMENTAÇÃO (COM DETECÇÃO AUTOMÁTICA DE POLARIDADE)
# =============================================================================

def estimar_polaridade(
    cinza: np.ndarray, mascara_padding: np.ndarray, cfg: Dict[str, Any]
) -> str:
    """
    Retorna 'escuro' (objeto mais escuro que o fundo) ou 'claro' (objeto mais claro).
    Quando cfg['polaridade'] == 'auto', estima pela mediana da faixa de borda da imagem
    (ignorando pixels de padding de rotação) comparada ao limiar de Otsu.
    """
    pol = str(cfg["polaridade"])
    if pol in ("escuro", "claro"):
        return pol
    if pol != "auto":
        raise ValueError(f"polaridade inválida: {pol}")

    h, w = cinza.shape
    bw = max(2, int(cfg["borda_largura"]))
    mask_borda = np.zeros((h, w), dtype=bool)
    mask_borda[:bw, :] = True
    mask_borda[-bw:, :] = True
    mask_borda[:, :bw] = True
    mask_borda[:, -bw:] = True

    borda_valida = mask_borda & (~mascara_padding)
    if int(borda_valida.sum()) < 20:
        borda_valida = mask_borda

    med_borda = float(np.median(cinza[borda_valida]))
    validos = cinza[~mascara_padding]
    if validos.size == 0:
        validos = cinza.ravel()
    otsu_val, _ = cv2.threshold(validos.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return "escuro" if med_borda >= float(otsu_val) else "claro"


def etapa2_segmentar(
    cinza: np.ndarray, mascara_padding: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, str]:
    """Segmentação por Otsu ou limiarização adaptativa com polaridade automática."""
    polaridade = estimar_polaridade(cinza, mascara_padding, cfg)
    metodo = str(cfg["seg_metodo"])
    tipo = cv2.THRESH_BINARY_INV if polaridade == "escuro" else cv2.THRESH_BINARY

    if metodo == "otsu":
        validos = cinza[~mascara_padding]
        if validos.size == 0:
            validos = cinza.ravel()
        otsu_t, _ = cv2.threshold(validos.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t_final = float(np.clip(float(otsu_t) + float(cfg["otsu_offset"]), 1.0, 254.0))
        _, binaria = cv2.threshold(cinza, t_final, 255, tipo)
    elif metodo in ("adaptativo_gauss", "adaptativo_media"):
        adapt = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if metodo == "adaptativo_gauss"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        bloco = _impar(max(3, int(cfg["seg_bloco"])))
        c_val = float(cfg["seg_c"])
        # Objetos claros usam THRESH_BINARY; o sinal de C se inverte para exigir pixel > média + |C|.
        c_efetivo = c_val if polaridade == "escuro" else -c_val
        binaria = cv2.adaptiveThreshold(cinza, 255, adapt, tipo, bloco, c_efetivo)
    else:
        raise ValueError(f"seg_metodo desconhecido: {metodo}")

    if mascara_padding.any():
        binaria[mascara_padding] = 0

    return binaria, polaridade


# =============================================================================
# ETAPA 3 — MORFOLOGIA (DUAS MÁSCARAS: COM E SEM FUROS PREENCHIDOS)
# =============================================================================

def _preencher_furos(
    mascara: np.ndarray, area_max_furo_px: Optional[float] = None
) -> np.ndarray:
    """
    Preenche buracos internos desconectados da borda da imagem.
    Se area_max_furo_px for informado (> 0), preenche apenas buracos menores que
    esse teto, evitando preencher o fundo preso entre objetos densamente encostados.
    """
    h, w = mascara.shape
    padded = cv2.copyMakeBorder(mascara, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood, (0, 0), 255)
    fundo_ext = padded[1:-1, 1:-1]
    furos = cv2.bitwise_not(fundo_ext)
    if area_max_furo_px is None or area_max_furo_px <= 0:
        return cv2.bitwise_or(mascara, furos)

    n_cc, cc_lbl, stats, _ = cv2.connectedComponentsWithStats(furos, connectivity=4)
    furos_filtrados = np.zeros_like(furos)
    for cid in range(1, n_cc):
        if float(stats[cid, cv2.CC_STAT_AREA]) <= area_max_furo_px:
            furos_filtrados[cc_lbl == cid] = 255
    return cv2.bitwise_or(mascara, furos_filtrados)


def etapa3_morfologia(
    binaria: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica abertura e fechamento configuráveis e retorna:
      (mascara_preenchida, mascara_sem_preencher).
    """
    m = binaria.copy()
    k_ab = _impar(int(cfg["morf_abertura_k"]))
    it_ab = int(cfg["morf_abertura_iter"])
    if k_ab >= 3 and it_ab > 0:
        ker_ab = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_ab, k_ab))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker_ab, iterations=it_ab)

    k_fe = _impar(int(cfg["morf_fechamento_k"]))
    it_fe = int(cfg["morf_fechamento_iter"])
    if k_fe >= 3 and it_fe > 0:
        ker_fe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fe, k_fe))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker_fe, iterations=it_fe)

    mascara_sem_preencher = m
    furo_max_rel = float(cfg.get("furo_max_area_rel", 0.0))
    area_max_furo_px = furo_max_rel * float(m.shape[0] * m.shape[1]) if furo_max_rel > 0 else None
    mascara_preenchida = _preencher_furos(mascara_sem_preencher, area_max_furo_px=area_max_furo_px)
    return mascara_preenchida, mascara_sem_preencher


# =============================================================================
# ETAPA 4 — SEPARAÇÃO POR TRANSFORMADA DE DISTÂNCIA + MARCADORES + WATERSHED
# =============================================================================

def etapa4_separar_watershed(
    mascara_preenchida: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa objetos encostados usando transformada de distância euclidiana,
    supressão de múltiplos picos em corpos alongados (parafusos) e watershed.

    Controle contra superdivisão de parafusos alongados:
      1. Estima o raio típico (meia-largura típica) dos objetos na imagem via
         percentil alto da transformada de distância no foreground.
      2. Suaviza o mapa de distância com sigma proporcional ao raio típico
         para fundir ondulações ao longo da rosca/haste de parafusos.
      3. Exige altura mínima de pico relativa ao raio típico (global) E ao
         máximo de distância de cada componente conectado (local).
      4. Exige distância mínima entre picos proporcional à largura típica
         (2 * raio_tipico * ws_dist_min_rel) via dilatação circular (máximo local 2D).
      5. Garante pelo menos 1 marcador por componente conectado válido para
         não perder objetos finos ou pequenos.

    Retorna (rotulos_ws, dist_suave, marcadores).
    """
    h, w = mascara_preenchida.shape
    dist = cv2.distanceTransform(mascara_preenchida, cv2.DIST_L2, 5)

    fg_vals = dist[dist > 0]
    if fg_vals.size == 0:
        vazio = np.zeros((h, w), dtype=np.int32)
        return vazio, dist, vazio

    raio_max_px = max(float(cfg["ws_raio_min_px"]), float(cfg["ws_raio_max_rel"]) * float(min(h, w)))
    raio_tipico = float(
        np.clip(
            float(np.percentile(fg_vals, float(cfg["ws_percentil_raio"]))),
            float(cfg["ws_raio_min_px"]),
            raio_max_px,
        )
    )

    sigma_dist = float(cfg["ws_suavizacao_fator"]) * raio_tipico
    if sigma_dist > 0.3:
        dist_suave = cv2.GaussianBlur(dist, (0, 0), sigmaX=sigma_dist, sigmaY=sigma_dist)
        dist_suave = dist_suave * (mascara_preenchida > 0)
    else:
        dist_suave = dist.copy()

    if not bool(cfg["ws_habilitado"]):
        n_cc, cc_labels = cv2.connectedComponents(mascara_preenchida, connectivity=8)
        return cc_labels.astype(np.int32), dist_suave, cc_labels.astype(np.int32)

    h_min_global = max(
        float(cfg["ws_pico_min_px"]),
        float(cfg["ws_altura_rel"]) * raio_tipico,
    )
    d_min = max(3, int(round(float(cfg["ws_dist_min_rel"]) * 2.0 * raio_tipico)))
    diam = _impar(2 * d_min + 1)
    # No OpenCV o kernel elíptico grande não é separável; acima de 31 px usa-se o retangular, que é.
    if diam <= 31:
        ker_max = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diam, diam))
    else:
        diam_rect = _impar(max(31, int(round(diam * 0.9))))
        ker_max = cv2.getStructuringElement(cv2.MORPH_RECT, (diam_rect, diam_rect))
    dil = cv2.dilate(dist_suave, ker_max)

    # Máximos locais estritos dentro do objeto e acima da altura mínima global.
    picos_mask = (
        (mascara_preenchida > 0)
        & (dist_suave == dil)
        & (dist_suave >= h_min_global)
    )

    # Refina por componente conectado: altura mínima relativa ao máximo do componente,
    # com ao menos 1 pico por componente de área razoável.
    n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(mascara_preenchida, connectivity=8)
    rel_local = float(cfg["ws_altura_local_rel"])
    area_min_px = max(4, int(round(float(cfg["area_min_rel"]) * h * w * 0.5)))

    sementes = np.zeros((h, w), dtype=np.uint8)
    for cid in range(1, n_cc):
        if int(cc_stats[cid, cv2.CC_STAT_AREA]) < area_min_px:
            continue
        x = int(cc_stats[cid, cv2.CC_STAT_LEFT])
        y = int(cc_stats[cid, cv2.CC_STAT_TOP])
        bw = int(cc_stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(cc_stats[cid, cv2.CC_STAT_HEIGHT])

        roi_lbl = cc_labels[y : y + bh, x : x + bw]
        comp_mask = roi_lbl == cid
        roi_dist = dist_suave[y : y + bh, x : x + bw]
        max_comp = float(roi_dist[comp_mask].max())
        if max_comp <= 0:
            continue
        roi_picos = picos_mask[y : y + bh, x : x + bw]
        picos_comp = roi_picos & comp_mask & (roi_dist >= rel_local * max_comp)
        roi_sem = sementes[y : y + bh, x : x + bw]
        if np.any(picos_comp):
            roi_sem[picos_comp] = 255
        else:
            # Sem pico válido, usa 1 marcador no ponto de maior distância do componente.
            ys, xs = np.where(comp_mask)
            vals = roi_dist[ys, xs]
            idx_max = int(np.argmax(vals))
            roi_sem[ys[idx_max], xs[idx_max]] = 255

    # Funde platôs e picos muito próximos (até ws_fusao_rel do raio típico) em uma única semente.
    k_fusao = _impar(max(3, int(round(float(cfg["ws_fusao_rel"]) * raio_tipico))))
    if k_fusao >= 3:
        ker_f = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_fusao, k_fusao))
        sementes = cv2.dilate(sementes, ker_f)
        sementes[mascara_preenchida == 0] = 0

    _, marc_fg = cv2.connectedComponents(sementes, connectivity=8)

    # Marcadores no formato do cv2.watershed:
    # 1 = fundo certo, >= 2 = objetos, 0 = região desconhecida (objeto sem semente).
    marcadores = np.zeros((h, w), dtype=np.int32)
    marcadores[mascara_preenchida == 0] = 1
    marcadores[marc_fg > 0] = marc_fg[marc_fg > 0] + 1

    # O watershed roda sobre a distância invertida, que é estável diante de texturas.
    d_max = float(dist_suave.max())
    if d_max > 0:
        topo = np.rint(255.0 * (1.0 - np.clip(dist_suave / d_max, 0.0, 1.0))).astype(np.uint8)
    else:
        topo = np.full((h, w), 255, dtype=np.uint8)
    topo_bgr = cv2.cvtColor(topo, cv2.COLOR_GRAY2BGR)

    rotulos_ws = cv2.watershed(topo_bgr, marcadores.copy())
    # Fica só com os rótulos de objeto (>= 2) dentro da máscara preenchida.
    rotulos_obj = np.where((rotulos_ws >= 2) & (mascara_preenchida > 0), rotulos_ws - 1, 0).astype(np.int32)
    return rotulos_obj, dist_suave, marcadores


# =============================================================================
# ETAPA 5 — COMPONENTES E DESCRITORES DE FORMA
# =============================================================================

def etapa5_componentes(
    rotulos_obj: np.ndarray,
    mascara_sem_preencher: np.ndarray,
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Aplica connectedComponentsWithStats sobre os objetos separados (preservando
    fronteiras do watershed), filtra por área relativa mínima/máxima e calcula
    os descritores de forma de cada componente.
    """
    h, w = rotulos_obj.shape
    area_img = float(h * w)
    area_min_px = float(cfg["area_min_rel"]) * area_img
    area_max_px = float(cfg["area_max_rel"]) * area_img
    poly_eps_fator = float(cfg["poly_eps_fator"])

    # Máscara binária com linha de corte de 1 px entre rótulos vizinhos diferentes.
    mascara_sep = (rotulos_obj > 0).astype(np.uint8) * 255
    if h > 1 and w > 1:
        borda_h = (rotulos_obj[:, :-1] != rotulos_obj[:, 1:]) & (rotulos_obj[:, :-1] > 0) & (rotulos_obj[:, 1:] > 0)
        borda_v = (rotulos_obj[:-1, :] != rotulos_obj[1:, :]) & (rotulos_obj[:-1, :] > 0) & (rotulos_obj[1:, :] > 0)
        mascara_sep[:, :-1][borda_h] = 0
        mascara_sep[:-1, :][borda_v] = 0

    n_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(mascara_sep, connectivity=8)

    componentes: List[Dict[str, Any]] = []
    for cid in range(1, n_cc):
        area = float(stats[cid, cv2.CC_STAT_AREA])
        if area < area_min_px or area > area_max_px:
            continue

        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        bw = int(stats[cid, cv2.CC_STAT_WIDTH])
        bh = int(stats[cid, cv2.CC_STAT_HEIGHT])

        # Recorte (ROI) do componente, para acelerar os cálculos geométricos.
        roi_comp = (labels_cc[y : y + bh, x : x + bw] == cid).astype(np.uint8) * 255
        roi_sem = cv2.bitwise_and(mascara_sem_preencher[y : y + bh, x : x + bw], roi_comp)

        contornos, _ = cv2.findContours(roi_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contornos:
            continue
        cnt = max(contornos, key=cv2.contourArea)
        area_cnt = float(cv2.contourArea(cnt))
        if area_cnt <= 0:
            area_cnt = area
        perim = float(cv2.arcLength(cnt, True))

        # Razão de aspecto pelo retângulo mínimo (minAreaRect).
        (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
        lado_maior = max(float(rw), float(rh), 1.0)
        lado_menor = max(min(float(rw), float(rh)), 1.0)
        razao_aspecto = lado_maior / lado_menor

        # Circularidade e solidez.
        circularidade = (4.0 * math.pi * area_cnt) / max(perim * perim, 1e-6)
        circularidade = float(min(circularidade, 1.5))
        casco = cv2.convexHull(cnt)
        area_casco = float(cv2.contourArea(casco))
        solidez = float(area_cnt / max(area_casco, 1e-6))

        # Aproximação poligonal, para separar contorno sextavado de redondo.
        approx = cv2.approxPolyDP(cnt, poly_eps_fator * perim, True)
        n_vertices = int(len(approx))

        # Presença e proporção de furo interno via máscara sem preenchimento + hierarquia
        # Considera buracos internos de roi_sem (pixels 0 dentro de roi_comp desconectados da borda)
        roi_preench_local = _preencher_furos(roi_sem)
        furo_mask = cv2.bitwise_and(roi_preench_local, cv2.bitwise_not(roi_sem))
        area_furo = float(np.count_nonzero(furo_mask))
        # Também confere diferença direta entre roi_comp e roi_sem restrita ao interior erodido
        if area_furo == 0.0:
            ker_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            interior = cv2.erode(roi_comp, ker_in, iterations=1)
            furo_int = cv2.bitwise_and(interior, cv2.bitwise_not(roi_sem))
            # Apenas componentes de furo desconectados da borda de roi_comp
            furo_cc = cv2.bitwise_and(furo_int, _preencher_furos(roi_sem))
            area_furo = float(np.count_nonzero(furo_cc))

        proporcao_furo = float(area_furo / max(area, 1.0))
        tem_furo = bool(proporcao_furo >= float(cfg["furo_min_prop_presenca"]))

        componentes.append(
            {
                "id": cid,
                "bbox_px": (x, y, x + bw, y + bh),
                "bbox_norm": (x / w, y / h, (x + bw) / w, (y + bh) / h),
                "centroide": (float(centroids[cid][0]), float(centroids[cid][1])),
                "area": area,
                "area_rel": float(area / area_img),
                "razao_aspecto": float(razao_aspecto),
                "tem_furo": tem_furo,
                "proporcao_furo": proporcao_furo,
                "circularidade": float(circularidade),
                "solidez": float(solidez),
                "n_vertices": n_vertices,
            }
        )

    return componentes


# =============================================================================
# ETAPA 6 — CLASSIFICAÇÃO POR LIMIARES LEGÍVEIS
# =============================================================================

def etapa6_classificar_componentes(
    componentes: Sequence[Dict[str, Any]],
    classes_dataset: Sequence[str],
    cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Classifica cada componente usando regras legíveis por limiares configuráveis
    (sem rede neural nem classificador treinado).
    """
    classes_set = set(classes_dataset)
    lim = cfg["limiares_classificacao"]
    saida: List[Dict[str, Any]] = []

    for comp in componentes:
        item = dict(comp)
        if len(classes_dataset) == 1:
            classe = classes_dataset[0]
        elif classes_set == {"parafuso", "porca"}:
            # Regra legível para parafuso x porca:
            # 1) Alongado (razão de aspecto alta) -> parafuso.
            # 2) Compacto com furo ou circularidade alta -> porca.
            # 3) Os demais casos se decidem pela razão de aspecto contra a área relativa.
            ar = comp["razao_aspecto"]
            furo = comp["proporcao_furo"]
            circ = comp["circularidade"]
            if ar >= float(lim["sn_aspecto_parafuso"]):
                classe = "parafuso"
            elif furo >= float(lim["sn_furo_porca"]) and ar < float(lim["sn_aspecto_max_porca"]):
                classe = "porca"
            elif circ >= float(lim["sn_circ_porca"]) and ar < float(lim["sn_aspecto_parafuso"]):
                classe = "porca"
            else:
                classe = "parafuso" if ar >= float(lim["sn_aspecto_desempate"]) else "porca"
        elif classes_set == {"parafuso", "porca", "arruela"}:
            # Regra legível para parafuso x porca x arruela:
            # 1) Parafuso é alongado (razão de aspecto alta) e/ou tem circularidade baixa.
            # 2) Entre os compactos (porca x arruela), a arruela tende a ter circularidade maior,
            #    mais vértices no approxPolyDP, área relativa maior e proporção de furo
            #    diferente da porca sextavada.
            ar = comp["razao_aspecto"]
            furo = comp["proporcao_furo"]
            circ = comp["circularidade"]
            sol = comp["solidez"]
            nv = comp["n_vertices"]
            area_rel = comp["area_rel"]

            if ar >= float(lim["nb_aspecto_parafuso"]):
                classe = "parafuso"
            elif circ < float(lim["nb_circ_min_compacto"]) and sol < float(lim["nb_solidez_min_compacto"]):
                classe = "parafuso"
            else:
                # Porca (sextavada, menor) x arruela (redonda, em anel).
                votos_arruela = 0
                if circ >= float(lim["nb_circ_arruela"]):
                    votos_arruela += 1
                if nv >= int(lim["nb_vertices_arruela"]):
                    votos_arruela += 1
                if area_rel >= float(lim["nb_area_rel_arruela"]):
                    votos_arruela += 1
                if furo >= float(lim["nb_furo_arruela"]):
                    votos_arruela += 1
                classe = "arruela" if votos_arruela >= int(lim["nb_votos_arruela"]) else "porca"
        else:
            raise ValueError(f"Conjunto de classes não suportado: {classes_dataset}")

        item["classe"] = classe
        saida.append(item)

    return saida


# =============================================================================
# ETAPA 7 — CONTAGEM POR CLASSE
# =============================================================================

def etapa7_contar_por_classe(
    componentes_classificados: Sequence[Dict[str, Any]],
    classes_dataset: Sequence[str],
) -> Dict[str, int]:
    """Conta os componentes por classe, incluindo as classes sem nenhum componente (contagem 0)."""
    cont = Counter(c["classe"] for c in componentes_classificados)
    return {cls: int(cont[cls]) for cls in classes_dataset}


def executar_pipeline_imagem(
    bgr: np.ndarray,
    classes_dataset: Sequence[str],
    cfg: Dict[str, Any],
    retornar_intermediarios: bool = False,
) -> Dict[str, Any]:
    """Executa as 7 etapas na ordem sobre uma imagem BGR."""
    proc, mascara_padding = etapa1_preprocessar(bgr, cfg)
    binaria, polaridade = etapa2_segmentar(proc, mascara_padding, cfg)
    masc_preench, masc_sem = etapa3_morfologia(binaria, cfg)
    rotulos_obj, dist_suave, marcadores = etapa4_separar_watershed(masc_preench, cfg)
    componentes = etapa5_componentes(rotulos_obj, masc_sem, cfg)
    classificados = etapa6_classificar_componentes(componentes, classes_dataset, cfg)
    contagens = etapa7_contar_por_classe(classificados, classes_dataset)

    res: Dict[str, Any] = {
        "contagens": contagens,
        "total_objetos": len(classificados),
        "componentes": classificados,
        "polaridade": polaridade,
    }
    if retornar_intermediarios:
        res["intermediarios"] = {
            "original": bgr,
            "preprocessada": proc,
            "binaria": binaria,
            "mascara_preenchida": masc_preench,
            "mascara_sem_preencher": masc_sem,
            "distancia": dist_suave,
            "marcadores": marcadores,
            "rotulos_ws": rotulos_obj,
        }
    return res


# =============================================================================
# CASAMENTO COM GT POR IoU >= 0.5 E HISTOGRAMAS NO TRAIN
# =============================================================================

def calcular_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """IoU entre duas caixas no formato (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    uniao = area_a + area_b - inter
    return inter / max(uniao, 1e-9)


def ler_caixas_gt(caminho_lbl: Path) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Lê um label YOLO e devolve [(classe, (x1, y1, x2, y2))] com coordenadas normalizadas."""
    caixas: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for ln in caminho_lbl.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        cid_s, cx_s, cy_s, bw_s, bh_s = ln.split()
        cid = int(cid_s)
        cx, cy, bw, bh = float(cx_s), float(cy_s), float(bw_s), float(bh_s)
        caixas.append((ID_PARA_CLASSE[cid], (cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0)))
    return caixas


def coletar_descritores_train(
    configs_por_dataset: Dict[str, Dict[str, Any]], iou_min: float = 0.5
) -> pd.DataFrame:
    """Casa componentes extraídos no TRAIN com caixas GT (IoU >= 0.5) para calibração de limiares."""
    linhas: List[dict] = []
    for ds_id, classes_ds in DATASETS_CLASSES.items():
        cfg = configs_por_dataset[ds_id]
        img_dir = PASTA_BASE / ds_id / "train" / "images"
        lbl_dir = PASTA_BASE / ds_id / "train" / "labels"
        for ip in sorted(img_dir.iterdir()):
            if not ip.is_file():
                continue
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            res = executar_pipeline_imagem(bgr, classes_ds, cfg)
            gt_boxes = ler_caixas_gt(lbl_dir / Path(ip.name).with_suffix(".txt").name)
            usados_gt: set[int] = set()
            for comp in res["componentes"]:
                melhor_iou = 0.0
                melhor_idx = -1
                for gi, (cls_gt, gbox) in enumerate(gt_boxes):
                    if gi in usados_gt:
                        continue
                    iou = calcular_iou(comp["bbox_norm"], gbox)
                    if iou > melhor_iou:
                        melhor_iou = iou
                        melhor_idx = gi
                if melhor_idx >= 0 and melhor_iou >= iou_min:
                    usados_gt.add(melhor_idx)
                    linhas.append(
                        {
                            "dataset": ds_id,
                            "imagem": ip.name,
                            "classe_gt": gt_boxes[melhor_idx][0],
                            "iou": melhor_iou,
                            "razao_aspecto": comp["razao_aspecto"],
                            "proporcao_furo": comp["proporcao_furo"],
                            "tem_furo": int(comp["tem_furo"]),
                            "circularidade": comp["circularidade"],
                            "solidez": comp["solidez"],
                            "n_vertices": comp["n_vertices"],
                            "area_rel": comp["area_rel"],
                        }
                    )
    return pd.DataFrame(linhas)


def desenhar_histogramas_descritores(df_desc: pd.DataFrame, pasta_out: Path) -> List[Path]:
    """Desenha histogramas limpos em PNG (com OpenCV) das distribuições por classe no TRAIN."""
    pasta_out.mkdir(parents=True, exist_ok=True)
    descritores = [
        ("razao_aspecto", "Razao de aspecto (minAreaRect)", (1.0, 6.0), 25),
        ("proporcao_furo", "Proporcao de furo interno", (0.0, 0.35), 25),
        ("circularidade", "Circularidade (4*pi*A / P^2)", (0.0, 1.15), 25),
        ("solidez", "Solidez (A / A_casco)", (0.3, 1.02), 25),
        ("n_vertices", "Numero de vertices (approxPolyDP)", (2.5, 16.5), 14),
        ("area_rel", "Area relativa (A / A_img)", (0.0, 0.04), 25),
    ]

    gerados: List[Path] = []
    grupos = [("geral_train", df_desc)]
    for ds_id in ("nutsnbolts", "screws_and_nuts"):
        sub = df_desc[df_desc["dataset"] == ds_id]
        if not sub.empty:
            grupos.append((ds_id + "_train", sub))

    for titulo_grp, df_g in grupos:
        # Painel 2 x 3 de histogramas
        cell_w, cell_h = 480, 300
        canvas = np.full((cell_h * 2 + 50, cell_w * 3, 3), 255, dtype=np.uint8)
        cv2.putText(
            canvas,
            f"Distribuicao de descritores no TRAIN (IoU >= 0.5) - {titulo_grp} (n={len(df_g)})",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        classes_pres = sorted(df_g["classe_gt"].unique(), key=lambda c: CLASSE_PARA_ID[c])
        # Legenda no topo direito
        lx = cell_w * 3 - 450
        for cls in classes_pres:
            n_c = int((df_g["classe_gt"] == cls).sum())
            cor = CORES_BGR[cls]
            cv2.rectangle(canvas, (lx, 14), (lx + 14, 28), cor, -1)
            cv2.putText(
                canvas,
                f"{cls} ({n_c})",
                (lx + 19, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
            lx += 145

        for idx, (col, label, (vmin, vmax), nbins) in enumerate(descritores):
            row, col_i = divmod(idx, 3)
            ox = col_i * cell_w
            oy = 50 + row * cell_h
            # Área de plotagem
            px1, py1 = ox + 55, oy + 36
            px2, py2 = ox + cell_w - 22, oy + cell_h - 42
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (245, 245, 245), -1)
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (180, 180, 180), 1)
            cv2.putText(
                canvas,
                label,
                (px1, oy + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            edges = np.linspace(vmin, vmax, nbins + 1)
            hists = {}
            max_dens = 1e-6
            for cls in classes_pres:
                vals = np.clip(df_g.loc[df_g["classe_gt"] == cls, col].to_numpy(dtype=float), vmin, vmax)
                if vals.size == 0:
                    continue
                cnts, _ = np.histogram(vals, bins=edges)
                dens = cnts / max(float(vals.size), 1.0)
                hists[cls] = dens
                max_dens = max(max_dens, float(dens.max()))

            plot_w = px2 - px1
            plot_h = py2 - py1
            bin_w = plot_w / nbins

            # Desenha barras semitransparentes e linhas de perfil
            overlay = canvas.copy()
            for cls in classes_pres:
                if cls not in hists:
                    continue
                cor = CORES_BGR[cls]
                dens = hists[cls]
                pts = []
                for b in range(nbins):
                    bx1 = int(round(px1 + b * bin_w))
                    bx2 = int(round(px1 + (b + 1) * bin_w)) - 1
                    bh = int(round((dens[b] / max_dens) * (plot_h - 10)))
                    if bh > 0:
                        cv2.rectangle(overlay, (bx1, py2 - bh), (max(bx1, bx2), py2), cor, -1)
                    pts.append((int(round((bx1 + bx2) / 2)), py2 - bh))
                if len(pts) >= 2:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, cor, 2, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0, canvas)

            # Ticks eixo X
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                tx = int(round(px1 + frac * plot_w))
                val_tick = vmin + frac * (vmax - vmin)
                cv2.line(canvas, (tx, py2), (tx, py2 + 4), (90, 90, 90), 1)
                cv2.putText(
                    canvas,
                    f"{val_tick:.2f}" if vmax <= 2.0 else f"{val_tick:.1f}",
                    (tx - 14, py2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (60, 60, 60),
                    1,
                    cv2.LINE_AA,
                )

            # Mediana por classe no canto do subplot
            my = py1 + 16
            for cls in classes_pres:
                vals = df_g.loc[df_g["classe_gt"] == cls, col].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                txt_med = f"{cls[:3]} med={np.median(vals):.3f}" if vmax <= 0.5 else f"{cls[:3]} med={np.median(vals):.2f}"
                cv2.putText(
                    canvas,
                    txt_med,
                    (px2 - 115, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    CORES_BGR[cls],
                    1,
                    cv2.LINE_AA,
                )
                my += 15

        out_path = pasta_out / f"histogramas_{titulo_grp}.png"
        cv2.imwrite(str(out_path), canvas)
        gerados.append(out_path)

    return gerados


# =============================================================================
# FIGURAS DAS ETAPAS (7 ETAPAS PARA 2 IMAGENS POR DATASET)
# =============================================================================

def gerar_figuras_etapas(
    configs_por_dataset: Dict[str, Dict[str, Any]],
    split: str = "test",
    n_por_dataset: int = 2,
) -> List[Path]:
    """Salva em PASTA_FIGURAS um painel com as 7 etapas para 2 imagens de cada dataset."""
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    geradas: List[Path] = []

    for ds_id, classes_ds in DATASETS_CLASSES.items():
        cfg = configs_por_dataset[ds_id]
        img_dir = PASTA_BASE / ds_id / split / "images"
        lbl_dir = PASTA_BASE / ds_id / split / "labels"
        imgs = sorted(p for p in img_dir.iterdir() if p.is_file())
        escolhidas = [imgs[0], imgs[len(imgs) // 2]] if len(imgs) >= 2 else imgs[:n_por_dataset]

        for idx_img, ip in enumerate(escolhidas, start=1):
            bgr = cv2.imread(str(ip), cv2.IMREAD_COLOR)
            res = executar_pipeline_imagem(bgr, classes_ds, cfg, retornar_intermediarios=True)
            inter = res["intermediarios"]
            gt_boxes = ler_caixas_gt(lbl_dir / Path(ip.name).with_suffix(".txt").name)
            gt_cnt = Counter(c for c, _ in gt_boxes)

            # Todas as etapas viram BGR do mesmo tamanho para entrar no painel.
            h, w = bgr.shape[:2]
            tw, th = 320, int(round(320 * h / w))

            def _fit(im: np.ndarray, titulo: str) -> np.ndarray:
                if im.ndim == 2:
                    im_bgr = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                else:
                    im_bgr = im.copy()
                rs = cv2.resize(im_bgr, (tw, th), interpolation=cv2.INTER_AREA)
                faixa = cv2.copyMakeBorder(rs, 24, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
                cv2.putText(faixa, titulo, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
                return faixa

            # Mapa de distância em falsa cor.
            d = inter["distancia"]
            d_norm = np.rint(255.0 * d / max(float(d.max()), 1e-6)).astype(np.uint8)
            dist_vis = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

            # Marcadores desenhados sobre a máscara preenchida.
            marc = inter["marcadores"]
            marc_vis = cv2.cvtColor(inter["mascara_preenchida"], cv2.COLOR_GRAY2BGR)
            marc_vis[marc >= 2] = (0, 0, 255)

            # Watershed com uma cor por rótulo.
            ws = inter["rotulos_ws"]
            ws_vis = np.zeros_like(bgr)
            rng_cor = np.random.default_rng(123)
            max_lbl = int(ws.max())
            if max_lbl > 0:
                paleta = rng_cor.integers(50, 255, size=(max_lbl + 1, 3), dtype=np.uint8)
                paleta[0] = (0, 0, 0)
                ws_vis = paleta[ws]

            # Resultado final com caixas e classes
            final_vis = bgr.copy()
            for comp in res["componentes"]:
                x1, y1, x2, y2 = comp["bbox_px"]
                cls = comp["classe"]
                cor = CORES_BGR[cls]
                cv2.rectangle(final_vis, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(
                    final_vis,
                    cls,
                    (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    cor,
                    1,
                    cv2.LINE_AA,
                )

            # Quadro resumo (8ª célula do grid 2x4)
            quadro = np.full((th + 24, tw, 3), 245, dtype=np.uint8)
            cv2.rectangle(quadro, (0, 0), (tw, 24), (28, 28, 28), -1)
            cv2.putText(quadro, "Resumo Contagem vs GT", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            y_cur = 46
            cv2.putText(quadro, f"Dataset: {ds_id}", (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
            y_cur += 22
            cv2.putText(quadro, f"Polaridade: {res['polaridade']}", (10, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (20, 20, 20), 1, cv2.LINE_AA)
            y_cur += 26
            for cls in classes_ds:
                pred_v = res["contagens"][cls]
                gt_v = gt_cnt[cls]
                cv2.putText(
                    quadro,
                    f"{cls}: prev={pred_v} | gt={gt_v} (err={pred_v - gt_v:+d})",
                    (10, y_cur),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    CORES_BGR[cls],
                    1,
                    cv2.LINE_AA,
                )
                y_cur += 22
            cv2.putText(
                quadro,
                f"TOTAL: prev={res['total_objetos']} | gt={sum(gt_cnt.values())}",
                (10, y_cur + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (10, 10, 10),
                1,
                cv2.LINE_AA,
            )

            paineis = [
                _fit(inter["original"], "1. Original"),
                _fit(inter["preprocessada"], "2. Pre-processada"),
                _fit(inter["binaria"], "3. Binaria (seg)"),
                _fit(inter["mascara_preenchida"], "4. Morfologia (preench.)"),
                _fit(dist_vis, "5. Transf. Distancia"),
                _fit(marc_vis, "6. Marcadores"),
                _fit(ws_vis, "7. Watershed"),
                _fit(final_vis, "8. Resultado Final"),
            ]
            linha1 = np.hstack(paineis[:4])
            linha2 = np.hstack(paineis[4:])
            grade = np.vstack([linha1, linha2])
            # Barra superior com nome do arquivo e contagem
            cab = f"{ds_id} [{split}] {ip.name} | prev={res['contagens']} | gt={ {c: gt_cnt[c] for c in classes_ds} }"
            grade = cv2.copyMakeBorder(grade, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
            cv2.putText(grade, cab[:125], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)

            out_fig = PASTA_FIGURAS / f"{ds_id}_{split}_amostra{idx_img}.png"
            cv2.imwrite(str(out_fig), grade)
            geradas.append(out_fig)

    return geradas


# =============================================================================
# INFERÊNCIA SOBRE UMA DIVISÃO E DIAGNÓSTICO DO TOTAL
# =============================================================================

def inferir_split(
    configs_por_dataset: Dict[str, Dict[str, Any]],
    split: str,
    df_gt: pd.DataFrame,
    metodo: str = "classico",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Roda a pipeline em todas as imagens da divisão e devolve:
      - df_previsoes (formato do scripts/avaliar.py, só com as chaves presentes no GT)
      - df_total (total de objetos detectados, sem olhar a classe, contra a soma do GT)
    """
    gt_sp = df_gt[df_gt["split"] == split].copy()
    gt_soma_img = (
        gt_sp.groupby(["dataset", "split", "imagem"])["contagem"].sum().to_dict()
    )

    linhas_prev: List[dict] = []
    linhas_tot: List[dict] = []

    for ds_id in sorted(DATASETS_CLASSES):
        classes_ds = DATASETS_CLASSES[ds_id]
        cfg = configs_por_dataset[ds_id]
        imagens_gt = sorted(gt_sp.loc[gt_sp["dataset"] == ds_id, "imagem"].unique())
        img_dir = PASTA_BASE / ds_id / split / "images"

        for nome_img in imagens_gt:
            bgr = cv2.imread(str(img_dir / nome_img), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Imagem ilegível: {img_dir / nome_img}")
            res = executar_pipeline_imagem(bgr, classes_ds, cfg)
            for cls in classes_ds:
                linhas_prev.append(
                    {
                        "metodo": metodo,
                        "dataset": ds_id,
                        "split": split,
                        "imagem": nome_img,
                        "classe": cls,
                        "contagem": int(res["contagens"][cls]),
                    }
                )
            tot_gt = int(gt_soma_img[(ds_id, split, nome_img)])
            tot_prev = int(res["total_objetos"])
            linhas_tot.append(
                {
                    "dataset": ds_id,
                    "split": split,
                    "imagem": nome_img,
                    "total_gt": tot_gt,
                    "total_previsto": tot_prev,
                    "erro_total": tot_prev - tot_gt,
                    "erro_abs_total": abs(tot_prev - tot_gt),
                }
            )

    df_prev = pd.DataFrame(
        linhas_prev, columns=["metodo", "dataset", "split", "imagem", "classe", "contagem"]
    )
    df_tot = pd.DataFrame(
        linhas_tot,
        columns=[
            "dataset",
            "split",
            "imagem",
            "total_gt",
            "total_previsto",
            "erro_total",
            "erro_abs_total",
        ],
    )
    return df_prev, df_tot


# =============================================================================
# CALIBRAÇÃO (SOMENTE TRAIN E VALID — NUNCA TEST!)
# =============================================================================

def config_base_padrao() -> Dict[str, Any]:
    """Configuração de partida de todas as etapas; a calibração sobrescreve campos por dataset."""
    return {
        "mediana_k": 3,
        "ignorar_padding_preto": True,
        "padding_limiar": 6,
        "iluminacao_metodo": "nenhum",
        "iluminacao_k": 31,
        "iluminacao_sigma": 35.0,
        "iluminacao_intensidade": 0.65,
        "iluminacao_ganho_min": 0.85,
        "iluminacao_ganho_max": 1.15,
        "usar_clahe": False,
        "clahe_clip": 2.0,
        "clahe_grid": 8,
        "polaridade": "auto",
        "borda_largura": 6,
        "seg_metodo": "otsu",
        "otsu_offset": 0.0,
        "seg_bloco": 35,
        "seg_c": 6.0,
        "morf_abertura_k": 3,
        "morf_abertura_iter": 1,
        "morf_fechamento_k": 3,
        "morf_fechamento_iter": 1,
        "furo_max_area_rel": 0.025,
        "ws_habilitado": True,
        "ws_percentil_raio": 85.0,
        "ws_raio_min_px": 4.0,
        "ws_raio_max_rel": 0.08,
        "ws_suavizacao_fator": 0.35,
        "ws_pico_min_px": 2.5,
        "ws_altura_rel": 0.35,
        "ws_altura_local_rel": 0.55,
        "ws_dist_min_rel": 0.65,
        "ws_fusao_rel": 0.25,
        "area_min_rel": 0.0008,
        "area_max_rel": 0.35,
        "poly_eps_fator": 0.03,
        "furo_min_prop_presenca": 0.015,
        "limiares_classificacao": {
            "sn_aspecto_parafuso": 1.75,
            "sn_aspecto_max_porca": 1.65,
            "sn_aspecto_desempate": 1.55,
            "sn_furo_porca": 0.02,
            "sn_circ_porca": 0.72,
            "nb_aspecto_parafuso": 1.60,
            "nb_circ_min_compacto": 0.55,
            "nb_solidez_min_compacto": 0.80,
            "nb_circ_arruela": 0.78,
            "nb_vertices_arruela": 7,
            "nb_area_rel_arruela": 0.0045,
            "nb_furo_arruela": 0.04,
            "nb_votos_arruela": 2,
        },
    }


def _avaliar_mae_total_valid(
    ds_id: str, cfg: Dict[str, Any], imgs_valid: List[Tuple[str, np.ndarray, int]]
) -> float:
    """MAE da contagem total no valid (usado na busca em grade de segmentação/watershed)."""
    classes_ds = DATASETS_CLASSES[ds_id]
    erros = []
    for _, bgr, tot_gt in imgs_valid:
        res = executar_pipeline_imagem(bgr, classes_ds, cfg)
        erros.append(abs(res["total_objetos"] - tot_gt))
    return float(np.mean(erros))


def _avaliar_mae_classes_componentes(
    ds_id: str,
    cfg: Dict[str, Any],
    comps_valid: List[Tuple[str, List[Dict[str, Any]], Dict[str, int]]],
) -> float:
    """MAE médio por classe no valid a partir de componentes já extraídos (etapas 6 e 7)."""
    classes_ds = DATASETS_CLASSES[ds_id]
    erros = []
    for _, comps, gt_cls in comps_valid:
        classificados = etapa6_classificar_componentes(comps, classes_ds, cfg)
        cont = etapa7_contar_por_classe(classificados, classes_ds)
        for cls in classes_ds:
            erros.append(abs(cont[cls] - gt_cls[cls]))
    return float(np.mean(erros))


def calibrar_todos_datasets(df_gt: pd.DataFrame) -> Dict[str, Any]:
    """
    Etapa de calibração em 2 fases (usando SOMENTE train e valid):
      Fase A: Grade pequena e explícita de segmentação, morfologia, área mínima e watershed
              por dataset (critério: MAE no valid).
      Fase B: Extração de descritores no TRAIN (casamento IoU >= 0.5 com GT), geração dos
              histogramas e calibração dos limiares legíveis de classificação para os
              datasets multiclasse (nutsnbolts e screws_and_nuts), validados pelo MAE
              via scripts/avaliar.py (--split valid).
    """
    gt_val = df_gt[df_gt["split"] == "valid"].copy()

    # Grades pequenas e explícitas por dataset (de 32 a 48 combinações cada).
    grades: Dict[str, List[Dict[str, Any]]] = {
        "washers_hardee": [
            {
                "iluminacao_metodo": il,
                "seg_metodo": sm,
                "seg_bloco": 41,
                "seg_c": 6.0,
                "morf_abertura_k": ka,
                "morf_fechamento_k": kf,
                "furo_max_area_rel": 0.02,
                "area_min_rel": amin,
                "ws_altura_rel": w_h,
                "ws_dist_min_rel": w_d,
            }
            for il, sm, ka, kf, amin, w_h, w_d in itertools.product(
                ["nenhum", "gauss_divisao"],
                ["otsu", "adaptativo_gauss"],
                [3, 5],
                [1, 3],
                [0.0008],
                [0.35],
                [0.40, 0.55],
            )
        ],
        "roboflow_screws": [
            {
                "iluminacao_metodo": il,
                "seg_metodo": sm,
                "seg_bloco": 45,
                "seg_c": sc,
                "morf_abertura_k": ka,
                "morf_fechamento_k": 5,
                "area_min_rel": amin,
                "ws_habilitado": ws_on,
                "ws_altura_rel": 0.45,
                "ws_dist_min_rel": 0.90,
                "ws_suavizacao_fator": 0.45,
            }
            for il, sm, sc, ka, amin, ws_on in itertools.product(
                ["nenhum", "gauss_divisao"],
                ["otsu", "adaptativo_gauss"],
                [6.0, 10.0],
                [3, 5],
                [0.0008, 0.0015],
                [True, False],
            )
            if not (sm == "otsu" and sc != 6.0)
        ],
        "nutsnbolts": [
            {
                "iluminacao_metodo": il,
                "seg_metodo": sm,
                "otsu_offset": off,
                "seg_bloco": 41,
                "seg_c": 8.0,
                "morf_abertura_k": ka,
                "morf_fechamento_k": 3,
                "area_min_rel": amin,
                "ws_altura_rel": w_h,
                "ws_dist_min_rel": w_d,
                "ws_suavizacao_fator": 0.40,
            }
            for il, sm, off, ka, amin, w_h, w_d in itertools.product(
                ["nenhum", "gauss_divisao"],
                ["otsu", "adaptativo_gauss"],
                [-10.0, 0.0],
                [3],
                [0.00025, 0.0005],
                [0.25, 0.40],
                [0.55, 0.75],
            )
            if not (sm == "adaptativo_gauss" and off != 0.0)
        ],
        "screws_and_nuts": [
            {
                "iluminacao_metodo": il,
                "usar_clahe": cl,
                "seg_metodo": sm,
                "seg_bloco": 45,
                "seg_c": sc,
                "morf_abertura_k": 3,
                "morf_fechamento_k": 5,
                "area_min_rel": amin,
                "ws_altura_rel": 0.35,
                "ws_dist_min_rel": w_d,
                "ws_suavizacao_fator": 0.40,
            }
            for il, cl, sm, sc, amin, w_d in itertools.product(
                ["nenhum", "gauss_divisao", "tophat"],
                [False, True],
                ["otsu", "adaptativo_gauss"],
                [6.0],
                [0.0004, 0.0010],
                [0.65, 0.85],
            )
            if not (il == "tophat" and cl)
        ],
    }

    melhores_configs: Dict[str, Dict[str, Any]] = {}
    resumo_busca: Dict[str, Any] = {}

    print("=== FASE A DA CALIBRAÇÃO: SEGMENTAÇÃO / WATERSHED NO VALID ===")
    for ds_id in sorted(DATASETS_CLASSES):
        sub_gt = gt_val[gt_val["dataset"] == ds_id]
        imgs_unicas = sorted(sub_gt["imagem"].unique())
        gt_tot = sub_gt.groupby("imagem")["contagem"].sum().to_dict()
        cache_imgs = []
        for nome_img in imgs_unicas:
            bgr = cv2.imread(str(PASTA_BASE / ds_id / "valid" / "images" / nome_img), cv2.IMREAD_COLOR)
            cache_imgs.append((nome_img, bgr, int(gt_tot[nome_img])))

        grade = grades[ds_id]
        melhor_mae = float("inf")
        melhor_cfg = None
        for comb in grade:
            cfg = config_base_padrao()
            cfg.update(comb)
            mae = _avaliar_mae_total_valid(ds_id, cfg, cache_imgs)
            if mae < melhor_mae:
                melhor_mae = mae
                melhor_cfg = cfg

        assert melhor_cfg is not None
        melhores_configs[ds_id] = melhor_cfg
        resumo_busca[ds_id] = {
            "combinacoes_testadas": len(grade),
            "melhor_mae_total_valid": round(melhor_mae, 4),
        }
        print(
            f"  [{ds_id}] {len(grade)} combinações | melhor MAE total valid = {melhor_mae:.3f} | "
            f"seg={melhor_cfg['seg_metodo']}, ilum={melhor_cfg['iluminacao_metodo']}, "
            f"ka={melhor_cfg['morf_abertura_k']}, amin={melhor_cfg['area_min_rel']}, "
            f"ws={melhor_cfg['ws_habilitado']}(h={melhor_cfg['ws_altura_rel']}, d={melhor_cfg['ws_dist_min_rel']})"
        )

    # Fase B: Extrair descritores no TRAIN (IoU >= 0.5) e gerar histogramas
    print("\n=== FASE B DA CALIBRAÇÃO: DESCRITORES NO TRAIN (IoU >= 0.5) E HISTOGRAMAS ===")
    df_desc = coletar_descritores_train(melhores_configs, iou_min=0.5)
    PASTA_SAIDA_CLASSICO.mkdir(parents=True, exist_ok=True)
    df_desc.to_csv(PASTA_SAIDA_CLASSICO / "descritores_train.csv", index=False, encoding="utf-8")
    hists = desenhar_histogramas_descritores(df_desc, PASTA_HISTOGRAMAS)
    print(f"  Componentes casados no TRAIN (IoU >= 0.5): {len(df_desc)}")
    print(f"  Histogramas gerados: {[p.name for p in hists]}")

    estat_desc: Dict[str, Any] = {}
    for ds_id in ("nutsnbolts", "screws_and_nuts"):
        sub_d = df_desc[df_desc["dataset"] == ds_id]
        print(f"\n  Distribuição de descritores no TRAIN [{ds_id}] (n={len(sub_d)}):")
        estat_desc[ds_id] = {}
        for cls in DATASETS_CLASSES[ds_id]:
            sc = sub_d[sub_d["classe_gt"] == cls]
            if sc.empty:
                continue
            info_c = {
                "n": int(len(sc)),
                "razao_aspecto_med": round(float(sc["razao_aspecto"].median()), 3),
                "razao_aspecto_p25_p75": [
                    round(float(sc["razao_aspecto"].quantile(0.25)), 3),
                    round(float(sc["razao_aspecto"].quantile(0.75)), 3),
                ],
                "proporcao_furo_med": round(float(sc["proporcao_furo"].median()), 4),
                "tem_furo_pct": round(float(sc["tem_furo"].mean() * 100.0), 1),
                "circularidade_med": round(float(sc["circularidade"].median()), 3),
                "solidez_med": round(float(sc["solidez"].median()), 3),
                "n_vertices_med": round(float(sc["n_vertices"].median()), 2),
                "area_rel_med": round(float(sc["area_rel"].median()), 5),
            }
            estat_desc[ds_id][cls] = info_c
            print(f"    {cls}: {info_c}")

    # Deriva limiares candidatos a partir das medianas/quartis do TRAIN e refina no VALID
    # 1) screws_and_nuts (parafuso vs porca)
    sn_desc = df_desc[df_desc["dataset"] == "screws_and_nuts"]
    sn_par = sn_desc[sn_desc["classe_gt"] == "parafuso"]
    sn_por = sn_desc[sn_desc["classe_gt"] == "porca"]
    if not sn_par.empty and not sn_por.empty:
        ar_mid = 0.5 * (float(sn_par["razao_aspecto"].median()) + float(sn_por["razao_aspecto"].median()))
        circ_mid = 0.5 * (float(sn_par["circularidade"].median()) + float(sn_por["circularidade"].median()))
    else:
        ar_mid, circ_mid = 1.75, 0.70

    sub_gt_sn = gt_val[gt_val["dataset"] == "screws_and_nuts"]
    cache_sn = []
    for nome_img in sorted(sub_gt_sn["imagem"].unique()):
        bgr = cv2.imread(str(PASTA_BASE / "screws_and_nuts" / "valid" / "images" / nome_img), cv2.IMREAD_COLOR)
        gt_c = sub_gt_sn[sub_gt_sn["imagem"] == nome_img].set_index("classe")["contagem"].to_dict()
        res0 = executar_pipeline_imagem(bgr, DATASETS_CLASSES["screws_and_nuts"], melhores_configs["screws_and_nuts"])
        cache_sn.append((nome_img, res0["componentes"], {c: int(gt_c[c]) for c in DATASETS_CLASSES["screws_and_nuts"]}))

    melhor_mae_sn = float("inf")
    melhor_lim_sn = None
    for ar_t, c_t, f_t in itertools.product(
        sorted({round(ar_mid * k, 2) for k in (0.80, 0.90, 1.0, 1.10)}),
        sorted({round(circ_mid * k, 2) for k in (0.90, 1.0, 1.08)}),
        [0.015, 0.03],
    ):
        cfg_t = json.loads(json.dumps(melhores_configs["screws_and_nuts"]))
        cfg_t["limiares_classificacao"]["sn_aspecto_parafuso"] = ar_t
        cfg_t["limiares_classificacao"]["sn_aspecto_max_porca"] = round(ar_t * 0.95, 2)
        cfg_t["limiares_classificacao"]["sn_aspecto_desempate"] = round(ar_t * 0.90, 2)
        cfg_t["limiares_classificacao"]["sn_circ_porca"] = c_t
        cfg_t["limiares_classificacao"]["sn_furo_porca"] = f_t
        mae_c = _avaliar_mae_classes_componentes("screws_and_nuts", cfg_t, cache_sn)
        if mae_c < melhor_mae_sn:
            melhor_mae_sn = mae_c
            melhor_lim_sn = cfg_t["limiares_classificacao"]
    if melhor_lim_sn is not None:
        melhores_configs["screws_and_nuts"]["limiares_classificacao"] = melhor_lim_sn
        print(f"\n  [screws_and_nuts] Melhor MAE por classe no valid = {melhor_mae_sn:.3f} | limiares={melhor_lim_sn}", flush=True)

    # 2) nutsnbolts (parafuso vs porca vs arruela)
    nb_desc = df_desc[df_desc["dataset"] == "nutsnbolts"]
    nb_par = nb_desc[nb_desc["classe_gt"] == "parafuso"]
    nb_por = nb_desc[nb_desc["classe_gt"] == "porca"]
    nb_arr = nb_desc[nb_desc["classe_gt"] == "arruela"]
    ar_nb_mid = (
        0.5 * (float(nb_par["razao_aspecto"].median()) + float(nb_por["razao_aspecto"].median()))
        if not nb_par.empty and not nb_por.empty
        else 1.60
    )
    area_nb_mid = (
        0.5 * (float(nb_arr["area_rel"].median()) + float(nb_por["area_rel"].median()))
        if not nb_arr.empty and not nb_por.empty
        else 0.0045
    )
    circ_nb_mid = (
        0.5 * (float(nb_arr["circularidade"].median()) + float(nb_por["circularidade"].median()))
        if not nb_arr.empty and not nb_por.empty
        else 0.78
    )

    sub_gt_nb = gt_val[gt_val["dataset"] == "nutsnbolts"]
    cache_nb = []
    for nome_img in sorted(sub_gt_nb["imagem"].unique()):
        bgr = cv2.imread(str(PASTA_BASE / "nutsnbolts" / "valid" / "images" / nome_img), cv2.IMREAD_COLOR)
        gt_c = sub_gt_nb[sub_gt_nb["imagem"] == nome_img].set_index("classe")["contagem"].to_dict()
        res0 = executar_pipeline_imagem(bgr, DATASETS_CLASSES["nutsnbolts"], melhores_configs["nutsnbolts"])
        cache_nb.append((nome_img, res0["componentes"], {c: int(gt_c[c]) for c in DATASETS_CLASSES["nutsnbolts"]}))

    melhor_mae_nb = float("inf")
    melhor_lim_nb = None
    for ar_t, area_t, circ_t, nv_t, votos_t in itertools.product(
        sorted({round(ar_nb_mid * k, 2) for k in (0.80, 0.90, 1.0, 1.15)}),
        sorted({round(area_nb_mid * k, 5) for k in (0.70, 1.0, 1.30)}),
        [round(circ_nb_mid, 2), round(min(0.90, circ_nb_mid + 0.05), 2)],
        [7, 8],
        [2, 3],
    ):
        cfg_t = json.loads(json.dumps(melhores_configs["nutsnbolts"]))
        cfg_t["limiares_classificacao"]["nb_aspecto_parafuso"] = ar_t
        cfg_t["limiares_classificacao"]["nb_area_rel_arruela"] = area_t
        cfg_t["limiares_classificacao"]["nb_circ_arruela"] = circ_t
        cfg_t["limiares_classificacao"]["nb_vertices_arruela"] = nv_t
        cfg_t["limiares_classificacao"]["nb_votos_arruela"] = votos_t
        mae_c = _avaliar_mae_classes_componentes("nutsnbolts", cfg_t, cache_nb)
        if mae_c < melhor_mae_nb:
            melhor_mae_nb = mae_c
            melhor_lim_nb = cfg_t["limiares_classificacao"]
    if melhor_lim_nb is not None:
        melhores_configs["nutsnbolts"]["limiares_classificacao"] = melhor_lim_nb
        print(f"  [nutsnbolts] Melhor MAE por classe no valid = {melhor_mae_nb:.3f} | limiares={melhor_lim_nb}", flush=True)

    config_final = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "metodo": "classico",
        "nota": "Congelado após calibração em train e valid. Split test não foi tocado.",
        "resumo_calibracao_valid": resumo_busca,
        "estatisticas_descritores_train": estat_desc,
        "datasets": melhores_configs,
    }
    ARQUIVO_CONFIG_FINAL.write_text(
        json.dumps(config_final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    hash_cfg = sha256_arquivo(ARQUIVO_CONFIG_FINAL)
    print(f"\n[CONGELADO] {ARQUIVO_CONFIG_FINAL}")
    print(f"[SHA-256 config_final.json] {hash_cfg}")
    return config_final


# =============================================================================
# CLI PRINCIPAL
# =============================================================================

def main() -> None:
    """Lê --modo (calibrar, avaliar-valid ou avaliar-test) e executa a pipeline clássica correspondente."""
    parser = argparse.ArgumentParser(description="Fase 3 — Pipeline clássica de contagem.")
    parser.add_argument(
        "--modo",
        choices=("calibrar", "avaliar-valid", "avaliar-test"),
        required=True,
    )
    parser.add_argument(
        "--hash-esperado",
        type=str,
        default=None,
        help="SHA-256 congelado de resultados/classico/config_final.json (obrigatório em avaliar-test).",
    )
    args = parser.parse_args()
    df_gt = av.carregar_gt(ARQUIVO_GT)
    PASTA_SAIDA_CLASSICO.mkdir(parents=True, exist_ok=True)

    if args.modo == "calibrar":
        cfg_final = calibrar_todos_datasets(df_gt)
        df_prev_val, df_tot_val = inferir_split(cfg_final["datasets"], "valid", df_gt)
        prev_val_path = PASTA_SAIDA_CLASSICO / "previsoes_valid.csv"
        df_prev_val.to_csv(prev_val_path, index=False, encoding="utf-8")
        df_tot_val.to_csv(PASTA_SAIDA_CLASSICO / "total_por_imagem_valid.csv", index=False, encoding="utf-8")
        print("\n=== AVALIAÇÃO OFICIAL NO VALID (scripts/avaliar.py) ===")
        av.avaliar(
            caminho_previsoes=prev_val_path,
            caminho_gt=ARQUIVO_GT,
            nome_execucao="classico_valid",
            split="valid",
            imprimir=True,
        )
        return

    if not ARQUIVO_CONFIG_FINAL.is_file():
        raise FileNotFoundError(f"config_final.json não encontrado: {ARQUIVO_CONFIG_FINAL}")
    hash_atual = sha256_arquivo(ARQUIVO_CONFIG_FINAL)
    cfg_final = json.loads(ARQUIVO_CONFIG_FINAL.read_text(encoding="utf-8"))

    if args.modo == "avaliar-valid":
        df_prev_val, df_tot_val = inferir_split(cfg_final["datasets"], "valid", df_gt)
        prev_val_path = PASTA_SAIDA_CLASSICO / "previsoes_valid.csv"
        df_prev_val.to_csv(prev_val_path, index=False, encoding="utf-8")
        df_tot_val.to_csv(PASTA_SAIDA_CLASSICO / "total_por_imagem_valid.csv", index=False, encoding="utf-8")
        av.avaliar(
            caminho_previsoes=prev_val_path,
            caminho_gt=ARQUIVO_GT,
            nome_execucao="classico_valid",
            split="valid",
            imprimir=True,
        )
        return

    if args.modo == "avaliar-test":
        if not args.hash_esperado:
            raise ValueError("Modo avaliar-test exige --hash-esperado com o SHA-256 congelado de config_final.json.")
        if hash_atual != args.hash_esperado:
            raise ValueError(
                f"SHA-256 de config_final.json divergiu! Esperado={args.hash_esperado}, atual={hash_atual}"
            )
        print(f"[OK] SHA-256 de config_final.json conferido: {hash_atual}")

        df_prev_test, df_tot_test = inferir_split(cfg_final["datasets"], "test", df_gt)
        prev_test_path = PASTA_SAIDA_CLASSICO / "previsoes_test.csv"
        tot_test_path = PASTA_SAIDA_CLASSICO / "total_por_imagem.csv"
        df_prev_test.to_csv(prev_test_path, index=False, encoding="utf-8")
        df_tot_test.to_csv(tot_test_path, index=False, encoding="utf-8")

        figs = gerar_figuras_etapas(cfg_final["datasets"], split="test", n_por_dataset=2)
        print(f"[Figuras] Geradas {len(figs)} figuras em {PASTA_FIGURAS}")

        print("\n=== AVALIAÇÃO FINAL NO TEST (SEM EXCLUSÕES) ===")
        av.avaliar(
            caminho_previsoes=prev_test_path,
            caminho_gt=ARQUIVO_GT,
            nome_execucao="classico_test",
            split="test",
            imprimir=True,
        )

        print("\n=== AVALIAÇÃO FINAL NO TEST (COM exclusoes_teste.csv) ===")
        av.avaliar(
            caminho_previsoes=prev_test_path,
            caminho_gt=ARQUIVO_GT,
            nome_execucao="classico_test_com_exclusoes",
            split="test",
            caminho_exclusoes=ARQUIVO_EXCLUSOES_TEST,
            imprimir=True,
        )


if __name__ == "__main__":
    main()
