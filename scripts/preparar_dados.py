#!/usr/bin/env python3
"""Prepara a base final e o GT de contagem (Fase 1).

O que faz: lê os ZIPs dos quatro datasets em datasets/, unifica as classes em parafuso,
porca e arruela, agrupa imagens parecidas (nome-base da fonte e dHash com rotações e
espelhamento) e sorteia grupos inteiros para train, valid e test, com semente fixa,
para não haver vazamento entre as divisões. Depois grava a base, as pastas por
detector YOLO e o GT, valida tudo e compacta o resultado. A execução apaga e recria
data_final/ e data_final.zip, então dá o mesmo resultado em toda rodada.

Entradas: datasets/<arruelas|parafusos|porcas>/<dataset>/*.zip (ver DATASETS).
Saídas:
  - data_final/base/<dataset>/<divisão>/{images,labels}
  - data_final/yolo/<classe>/{train,valid,test}/{images,labels} e data.yaml
  - data_final/contagens_gt.csv (GT: dataset, split, imagem, classe, contagem)
  - data_final/manifesto.json, data_final/_amostras/ e data_final.zip

Como rodar:
    .venv/bin/python scripts/preparar_dados.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

# =============================================================================
# CONSTANTES E PARÂMETROS
# =============================================================================

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_SAIDA = RAIZ_PROJETO / "data_final"
ARQUIVO_ZIP_SAIDA = RAIZ_PROJETO / "data_final.zip"

SEMENTE = 42
PROPORCAO_TRAIN = 0.70
PROPORCAO_VALID = 0.15
PROPORCAO_TEST = 0.15

DHASH_TAMANHO = 8
DHASH_MAX_HAMMING = 4
LIMITE_MAX_FRACAO_GRUPO = 0.30

AMOSTRAS_POR_SPLIT = 2
EXTENSOES_IMAGEM = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS_SAIDA = ("train", "valid", "test")

CLASSES_FINAIS = ("parafuso", "porca", "arruela")
CLASSE_PARA_ID = {"parafuso": 0, "porca": 1, "arruela": 2}
ID_PARA_CLASSE = {v: k for k, v in CLASSE_PARA_ID.items()}

CORES_BGR = {
    0: (0, 165, 255),   # parafuso: laranja
    1: (255, 80, 0),    # porca: azul
    2: (0, 200, 0),     # arruela: verde
}


@dataclass(frozen=True)
class ConfigDataset:
    """Origem de um dataset: ZIP, mapa de nomes de classe e imagens a excluir pelo nome."""

    dataset_id: str
    caminho_zip: Path
    mapa_classes: Dict[str, str]
    exclusoes_substring: Tuple[str, ...] = ()


DATASETS: Tuple[ConfigDataset, ...] = (
    ConfigDataset(
        dataset_id="washers_hardee",
        caminho_zip=RAIZ_PROJETO / "datasets/arruelas/washers_hardee/deduplicado.zip",
        mapa_classes={"washer": "arruela"},
    ),
    ConfigDataset(
        dataset_id="roboflow_screws",
        caminho_zip=RAIZ_PROJETO / "datasets/parafusos/roboflow_screws/revisado.zip",
        mapa_classes={
            "Parafuso D2 4.0x25 C4": "parafuso",
            "Parafuso D3 35x30 c6": "parafuso",
            "Parafuso D4 4.5x16 C16": "parafuso",
            "parafuso D1 35x12 c2": "parafuso",
        },
        exclusoes_substring=("Parafusos_Dourados_0089",),
    ),
    ConfigDataset(
        dataset_id="nutsnbolts",
        caminho_zip=RAIZ_PROJETO / "datasets/porcas/nutsnbolts/filtrado.zip",
        mapa_classes={
            "bolt": "parafuso",
            "nut": "porca",
            "washer": "arruela",
        },
    ),
    ConfigDataset(
        dataset_id="screws_and_nuts",
        caminho_zip=RAIZ_PROJETO / "datasets/porcas/screws_and_nuts/original.zip",
        mapa_classes={
            "Screws": "parafuso",
            "Nuts": "porca",
        },
    ),
)


# =============================================================================
# UTILITÁRIOS
# =============================================================================

class UnionFind:
    """União-busca simples, usada para juntar imagens parecidas em grupos."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Devolve a raiz do conjunto de x, com compressão de caminho."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """Junta os conjuntos de a e b."""
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def sha256_arquivo(caminho: Path, bloco: int = 1024 * 1024) -> str:
    """SHA-256 de um arquivo, lido em blocos de 1 MiB."""
    h = hashlib.sha256()
    with caminho.open("rb") as f:
        for dados in iter(lambda: f.read(bloco), b""):
            h.update(dados)
    return h.hexdigest()


def nome_base_fonte(stem: str) -> str:
    """Remove sufixo Roboflow (.rf.<hex>) e sufixo de extensão (_jpg, _png, ...)."""
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", stem, flags=re.IGNORECASE)
    s = re.sub(r"_(jpg|jpeg|png|bmp|webp)$", "", s, flags=re.IGNORECASE)
    return s


def dhash_int(pil_img: Image.Image, tamanho: int = DHASH_TAMANHO) -> int:
    """dHash da imagem como inteiro: compara pixels vizinhos numa miniatura em tons de cinza."""
    resample = getattr(Image, "Resampling", Image).LANCZOS
    g = pil_img.convert("L").resize((tamanho + 1, tamanho), resample)
    arr = np.asarray(g)
    diff = (arr[:, 1:] > arr[:, :-1]).flatten()
    val = 0
    for b in diff:
        val = (val << 1) | int(bool(b))
    return val


def distancia_dhash_d4(
    h0_a: int, hd4_a: Tuple[int, ...], h0_b: int, hd4_b: Tuple[int, ...]
) -> int:
    """Distância de Hamming mínima entre uma imagem e as 8 variantes D4 (rotações e espelhos) da outra, nos dois sentidos."""
    d_ab = min((h0_a ^ hb).bit_count() for hb in hd4_b)
    d_ba = min((h0_b ^ ha).bit_count() for ha in hd4_a)
    return d_ab if d_ab < d_ba else d_ba


def dhash_d4_de_bytes(dados: bytes) -> Tuple[int, Tuple[int, ...]]:
    """Devolve o dHash da imagem original e os 8 dHashes das variantes D4."""
    with Image.open(io.BytesIO(dados)) as im:
        im.load()
        h0 = dhash_int(im)
        hashes: List[int] = []
        flip_lr = getattr(Image, "Transpose", Image).FLIP_LEFT_RIGHT
        for k in range(4):
            rot = im.rotate(90 * k) if k else im
            hashes.append(dhash_int(rot))
            hashes.append(dhash_int(rot.transpose(flip_lr)))
    return h0, tuple(hashes)


def carregar_classes_yaml(conteudo: bytes, dataset_id: str) -> Dict[int, str]:
    """Lê o campo names do data.yaml (lista ou dicionário) e devolve {id: nome}."""
    cfg = yaml.safe_load(conteudo.decode("utf-8-sig")) or {}
    nomes = cfg.get("names")
    if isinstance(nomes, list):
        return {i: str(n) for i, n in enumerate(nomes)}
    if isinstance(nomes, dict):
        return {int(k): str(v) for k, v in nomes.items()}
    raise ValueError(f"[{dataset_id}] data.yaml sem campo 'names' válido.")


def caminho_label_no_zip(caminho_img: str) -> str:
    """Troca a pasta images/ por labels/ e a extensão por .txt, dentro do ZIP."""
    partes = list(Path(caminho_img).parts)
    indices = [i for i, p in enumerate(partes) if p == "images"]
    if len(indices) != 1:
        raise ValueError(f"Caminho sem pasta 'images/' inequívoca: {caminho_img}")
    partes[indices[0]] = "labels"
    return Path(*partes).with_suffix(".txt").as_posix()


def converter_linha_para_bbox(
    linha: str,
    classes_orig: Dict[int, str],
    mapa_classes: Dict[str, str],
    contexto: str,
) -> Tuple[int, Tuple[float, float, float, float] | None, str]:
    """Converte uma linha de label (caixa com 5 valores ou polígono) em (id_unificado, caixa, tipo_origem).

    A caixa é (cx, cy, w, h) normalizada. Para polígonos vale o retângulo que os envolve.
    Se largura ou altura ficar menor ou igual a zero, a caixa volta como None.
    """
    tokens = linha.strip().split()
    if len(tokens) < 5:
        raise ValueError(f"{contexto}: linha com menos de 5 valores: '{linha}'")

    cid_orig = int(tokens[0])
    if cid_orig not in classes_orig:
        raise ValueError(f"{contexto}: ID de classe {cid_orig} ausente em data.yaml")

    nome_orig = classes_orig[cid_orig]
    if nome_orig not in mapa_classes:
        raise ValueError(f"{contexto}: classe '{nome_orig}' não mapeada")

    cid_final = CLASSE_PARA_ID[mapa_classes[nome_orig]]
    coords = [float(v) for v in tokens[1:]]
    if not all(math.isfinite(v) for v in coords):
        raise ValueError(f"{contexto}: coordenadas não finitas")

    if len(tokens) == 5:
        cx, cy, w, h = coords
        if w <= 0.0 or h <= 0.0:
            return cid_final, None, "caixa"
        return cid_final, (cx, cy, w, h), "caixa"

    if (len(tokens) - 1) % 2 != 0 or len(coords) < 6:
        raise ValueError(f"{contexto}: polígono malformado ({len(tokens)} tokens)")

    xs = [min(1.0, max(0.0, v)) for v in coords[0::2]]
    ys = [min(1.0, max(0.0, v)) for v in coords[1::2]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0.0 or h <= 0.0:
        return cid_final, None, "poligono"
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    return cid_final, (cx, cy, w, h), "poligono"


def formatar_bbox_yolo(cid: int, bbox: Tuple[float, float, float, float]) -> str:
    """Formata uma caixa como linha de label YOLO, com 6 casas decimais."""
    cx, cy, w, h = bbox
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# =============================================================================
# PROCESSAMENTO POR DATASET
# =============================================================================

@dataclass
class RegistroImagem:
    """Uma imagem já lida do ZIP: bytes, hashes e anotações convertidas para as classes finais."""

    dataset_id: str
    caminho_no_zip: str
    nome_original: str
    nome_saida: str
    bytes_img: bytes
    h0: int
    hd4: Tuple[int, ...]
    anotacoes: List[Tuple[int, Tuple[float, float, float, float]]]
    linhas_origem_por_classe: Counter
    descartadas_por_classe: Counter
    tipos_origem: Counter
    txt_vazio: bool


def carregar_dataset(cfg: ConfigDataset) -> Tuple[List[RegistroImagem], dict]:
    """Lê o ZIP do dataset, converte os labels e devolve (registros, metadados com o sha256 do ZIP)."""
    if not cfg.caminho_zip.is_file():
        raise FileNotFoundError(
            f"[{cfg.dataset_id}] ZIP de origem não encontrado: {cfg.caminho_zip}"
        )

    sha_zip = sha256_arquivo(cfg.caminho_zip)
    registros: List[RegistroImagem] = []
    imagens_excluidas: List[str] = []
    txts_vazios: List[str] = []
    nomes_saida_vistos: set[str] = set()

    with zipfile.ZipFile(cfg.caminho_zip, "r") as z:
        if z.testzip() is not None:
            raise ValueError(f"[{cfg.dataset_id}] ZIP corrompido: {cfg.caminho_zip}")

        nomes_zip = sorted({i.filename for i in z.infolist() if not i.is_dir()})
        yamls = [n for n in nomes_zip if Path(n).name == "data.yaml"]
        if len(yamls) != 1:
            raise ValueError(
                f"[{cfg.dataset_id}] Esperado exatamente 1 data.yaml, encontrados {len(yamls)}"
            )

        classes_orig = carregar_classes_yaml(z.read(yamls[0]), cfg.dataset_id)
        nao_mapeadas = [c for c in classes_orig.values() if c not in cfg.mapa_classes]
        if nao_mapeadas:
            raise ValueError(
                f"[{cfg.dataset_id}] Classes não mapeadas/ambíguas em data.yaml: {nao_mapeadas}"
            )

        nomes_set = set(nomes_zip)
        imagens = sorted(
            n
            for n in nomes_zip
            if "images" in Path(n).parts and Path(n).suffix.lower() in EXTENSOES_IMAGEM
        )
        if not imagens:
            raise ValueError(f"[{cfg.dataset_id}] Nenhuma imagem encontrada em images/")

        for img_zip in imagens:
            nome_orig = Path(img_zip).name
            if any(sub in nome_orig for sub in cfg.exclusoes_substring):
                imagens_excluidas.append(img_zip)
                continue

            lbl_zip = caminho_label_no_zip(img_zip)
            if lbl_zip not in nomes_set:
                raise ValueError(f"[{cfg.dataset_id}] Rótulo ausente para imagem: {img_zip}")

            nome_saida = f"{cfg.dataset_id}__{nome_orig}"
            if nome_saida in nomes_saida_vistos:
                raise ValueError(
                    f"[{cfg.dataset_id}] Colisão de nome de arquivo na saída: {nome_saida}"
                )
            nomes_saida_vistos.add(nome_saida)

            bytes_img = z.read(img_zip)
            h0, hd4 = dhash_d4_de_bytes(bytes_img)

            conteudo_lbl = z.read(lbl_zip).decode("utf-8-sig")
            linhas = [ln.strip() for ln in conteudo_lbl.splitlines() if ln.strip()]
            vazio = len(linhas) == 0
            if vazio:
                txts_vazios.append(img_zip)

            anotacoes: List[Tuple[int, Tuple[float, float, float, float]]] = []
            origem_cls: Counter = Counter()
            desc_cls: Counter = Counter()
            tipos: Counter = Counter()

            for idx_linha, ln in enumerate(linhas, start=1):
                ctx = f"{cfg.dataset_id}/{lbl_zip}:L{idx_linha}"
                cid_final, bbox, tipo = converter_linha_para_bbox(
                    ln, classes_orig, cfg.mapa_classes, ctx
                )
                origem_cls[cid_final] += 1
                tipos[tipo] += 1
                if bbox is None:
                    desc_cls[cid_final] += 1
                else:
                    anotacoes.append((cid_final, bbox))

            registros.append(
                RegistroImagem(
                    dataset_id=cfg.dataset_id,
                    caminho_no_zip=img_zip,
                    nome_original=nome_orig,
                    nome_saida=nome_saida,
                    bytes_img=bytes_img,
                    h0=h0,
                    hd4=hd4,
                    anotacoes=anotacoes,
                    linhas_origem_por_classe=origem_cls,
                    descartadas_por_classe=desc_cls,
                    tipos_origem=tipos,
                    txt_vazio=vazio,
                )
            )

    classes_rotuladas = sorted(
        {cfg.mapa_classes[c] for c in classes_orig.values()},
        key=lambda nome: CLASSE_PARA_ID[nome],
    )
    meta = {
        "dataset_id": cfg.dataset_id,
        "caminho_zip": str(cfg.caminho_zip.relative_to(RAIZ_PROJETO)),
        "sha256_zip": sha_zip,
        "classes_originais": {str(k): v for k, v in sorted(classes_orig.items())},
        "classes_rotuladas": classes_rotuladas,
        "imagens_excluidas": imagens_excluidas,
        "txts_vazios_nao_excluidos": txts_vazios,
    }
    return registros, meta


def agrupar_imagens(registros: Sequence[RegistroImagem], dataset_id: str) -> List[List[RegistroImagem]]:
    """Agrupa imagens com o mesmo nome-base de fonte ou dHash D4 a até 4 bits de distância.

    Aborta se um único grupo passar de 30% do dataset, o que indicaria encadeamento indevido.
    """
    n = len(registros)
    uf = UnionFind(n)

    por_base: Dict[str, List[int]] = defaultdict(list)
    for i, reg in enumerate(registros):
        chave = nome_base_fonte(Path(reg.nome_original).stem)
        por_base[chave].append(i)

    for indices in por_base.values():
        for j in indices[1:]:
            uf.union(indices[0], j)

    for i in range(n):
        hi, hd4_i = registros[i].h0, registros[i].hd4
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            if distancia_dhash_d4(hi, hd4_i, registros[j].h0, registros[j].hd4) <= DHASH_MAX_HAMMING:
                uf.union(i, j)

    grupos_map: Dict[int, List[RegistroImagem]] = defaultdict(list)
    for i, reg in enumerate(registros):
        grupos_map[uf.find(i)].append(reg)

    grupos: List[List[RegistroImagem]] = []
    for itens in grupos_map.values():
        itens.sort(key=lambda r: r.nome_original)
        grupos.append(itens)
    grupos.sort(key=lambda g: g[0].nome_original)

    maior = max(len(g) for g in grupos)
    fracao = maior / n
    if fracao > LIMITE_MAX_FRACAO_GRUPO:
        raise RuntimeError(
            f"[{dataset_id}] BLOQUEIO: maior grupo tem {maior}/{n} imagens "
            f"({fracao * 100:.2f}% > {LIMITE_MAX_FRACAO_GRUPO * 100:.0f}%). "
            "Provável encadeamento indevido."
        )
    return grupos


def dividir_grupos(
    grupos: Sequence[List[RegistroImagem]],
    classes_rotuladas: Sequence[str],
    dataset_id: str,
) -> Tuple[Dict[str, List[RegistroImagem]], Dict[str, int], int]:
    """Sorteia grupos inteiros para train, valid e test (70/15/15, semente fixa).

    Em valid e test fica só a primeira imagem de cada grupo; as demais são omitidas.
    Aborta se valid ou test ficar sem objetos de alguma classe do dataset.
    """
    n_grupos = len(grupos)
    if n_grupos < 3:
        raise RuntimeError(
            f"[{dataset_id}] Apenas {n_grupos} grupos; insuficiente para train/valid/test."
        )

    ordem = list(range(n_grupos))
    rng = random.Random(SEMENTE)
    rng.shuffle(ordem)

    n_valid = max(1, round(n_grupos * PROPORCAO_VALID))
    n_test = max(1, round(n_grupos * PROPORCAO_TEST))
    n_train = n_grupos - n_valid - n_test
    if n_train < 1:
        raise RuntimeError(f"[{dataset_id}] Divisão inválida de grupos: {n_train}/{n_valid}/{n_test}")

    grupos_por_split = {
        "train": [grupos[i] for i in ordem[:n_train]],
        "valid": [grupos[i] for i in ordem[n_train : n_train + n_valid]],
        "test": [grupos[i] for i in ordem[n_train + n_valid :]],
    }

    imagens_por_split: Dict[str, List[RegistroImagem]] = {"train": [], "valid": [], "test": []}
    omitidas_valid_test = 0

    for g in grupos_por_split["train"]:
        imagens_por_split["train"].extend(g)
    for sp in ("valid", "test"):
        for g in grupos_por_split[sp]:
            imagens_por_split[sp].append(g[0])  # a primeira por nome representa o grupo
            omitidas_valid_test += len(g) - 1

    for sp in SPLITS_SAIDA:
        imagens_por_split[sp].sort(key=lambda r: r.nome_saida)

    # valid e test precisam ter pelo menos um objeto de cada classe do dataset.
    for sp in ("valid", "test"):
        cont_sp: Counter = Counter()
        for reg in imagens_por_split[sp]:
            for cid, _ in reg.anotacoes:
                cont_sp[ID_PARA_CLASSE[cid]] += 1
        for cls in classes_rotuladas:
            if cont_sp[cls] == 0:
                raise RuntimeError(
                    f"[{dataset_id}] BLOQUEIO: divisão '{sp}' ficou com 0 objetos da classe '{cls}'."
                )

    contagem_grupos = {sp: len(grupos_por_split[sp]) for sp in SPLITS_SAIDA}
    return imagens_por_split, contagem_grupos, omitidas_valid_test


# =============================================================================
# ESCRITA DOS ARTEFATOS
# =============================================================================

def preparar_diretorios_saida() -> None:
    """Apaga data_final/ e data_final.zip de execuções anteriores e recria a pasta de saída."""
    if PASTA_SAIDA.exists():
        shutil.rmtree(PASTA_SAIDA)
    if ARQUIVO_ZIP_SAIDA.exists():
        ARQUIVO_ZIP_SAIDA.unlink()
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


def gravar_base_e_yolo(
    por_dataset: Dict[str, Dict[str, List[RegistroImagem]]],
    metas: Dict[str, dict],
) -> pd.DataFrame:
    """Grava base/, yolo/<classe>/ (um label por imagem, só com a classe do detector) e contagens_gt.csv."""
    linhas_gt: List[dict] = []

    # Cria yolo/<classe>/{train,valid,test}/{images,labels} antes de gravar.
    for cls in CLASSES_FINAIS:
        for sp in SPLITS_SAIDA:
            (PASTA_SAIDA / "yolo" / cls / sp / "images").mkdir(parents=True, exist_ok=True)
            (PASTA_SAIDA / "yolo" / cls / sp / "labels").mkdir(parents=True, exist_ok=True)

    for dataset_id, splits_dict in por_dataset.items():
        classes_rotuladas = metas[dataset_id]["classes_rotuladas"]
        for sp in SPLITS_SAIDA:
            base_img_dir = PASTA_SAIDA / "base" / dataset_id / sp / "images"
            base_lbl_dir = PASTA_SAIDA / "base" / dataset_id / sp / "labels"
            base_img_dir.mkdir(parents=True, exist_ok=True)
            base_lbl_dir.mkdir(parents=True, exist_ok=True)

            for reg in splits_dict[sp]:
                # 1. Imagem original, byte a byte, em base/.
                (base_img_dir / reg.nome_saida).write_bytes(reg.bytes_img)

                # 2. Label com as três classes unificadas, em base/.
                nome_txt = Path(reg.nome_saida).with_suffix(".txt").name
                linhas_base = [formatar_bbox_yolo(cid, bbox) for cid, bbox in reg.anotacoes]
                (base_lbl_dir / nome_txt).write_text(
                    ("\n".join(linhas_base) + "\n") if linhas_base else "",
                    encoding="utf-8",
                )

                # 3. Linha do GT para cada classe que o dataset rotula, inclusive com contagem 0.
                cont_img = Counter(ID_PARA_CLASSE[cid] for cid, _ in reg.anotacoes)
                for cls in classes_rotuladas:
                    linhas_gt.append(
                        {
                            "dataset": dataset_id,
                            "split": sp,
                            "imagem": reg.nome_saida,
                            "classe": cls,
                            "contagem": int(cont_img[cls]),
                        }
                    )

                # 4. Cópia em yolo/<classe>/ com classe 0 e só as caixas daquela classe.
                for cls in classes_rotuladas:
                    cid_alvo = CLASSE_PARA_ID[cls]
                    yolo_img_dir = PASTA_SAIDA / "yolo" / cls / sp / "images"
                    yolo_lbl_dir = PASTA_SAIDA / "yolo" / cls / sp / "labels"
                    (yolo_img_dir / reg.nome_saida).write_bytes(reg.bytes_img)
                    linhas_cls = [
                        formatar_bbox_yolo(0, bbox)
                        for cid, bbox in reg.anotacoes
                        if cid == cid_alvo
                    ]
                    (yolo_lbl_dir / nome_txt).write_text(
                        ("\n".join(linhas_cls) + "\n") if linhas_cls else "",
                        encoding="utf-8",
                    )

    # data.yaml de cada detector YOLO (uma classe só).
    for cls in CLASSES_FINAIS:
        yaml_path = PASTA_SAIDA / "yolo" / cls / "data.yaml"
        conteudo_yaml = {
            "path": ".",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 1,
            "names": [cls],
        }
        yaml_path.write_text(
            yaml.safe_dump(conteudo_yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    df_gt = pd.DataFrame(linhas_gt, columns=["dataset", "split", "imagem", "classe", "contagem"])
    df_gt.to_csv(PASTA_SAIDA / "contagens_gt.csv", index=False, encoding="utf-8")
    return df_gt


def gerar_amostras_visuais(por_dataset: Dict[str, Dict[str, List[RegistroImagem]]]) -> int:
    """Desenha caixas em 2 imagens sorteadas por dataset e divisão, para conferir os labels a olho."""
    pasta_amostras = PASTA_SAIDA / "_amostras"
    pasta_amostras.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEMENTE)
    total_geradas = 0

    for dataset_id, splits_dict in por_dataset.items():
        for sp in SPLITS_SAIDA:
            pasta_sp = pasta_amostras / dataset_id / sp
            pasta_sp.mkdir(parents=True, exist_ok=True)
            base_img_dir = PASTA_SAIDA / "base" / dataset_id / sp / "images"
            base_lbl_dir = PASTA_SAIDA / "base" / dataset_id / sp / "labels"

            candidatos = sorted(p.name for p in base_img_dir.iterdir() if p.is_file())
            escolhidas = rng.sample(candidatos, k=min(AMOSTRAS_POR_SPLIT, len(candidatos)))

            for nome_img in escolhidas:
                img = cv2.imread(str(base_img_dir / nome_img), cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"Falha ao ler imagem para amostra: {base_img_dir / nome_img}")
                h_img, w_img = img.shape[:2]
                lbl_path = base_lbl_dir / Path(nome_img).with_suffix(".txt").name
                for ln in lbl_path.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    cid_s, cx_s, cy_s, bw_s, bh_s = ln.split()
                    cid = int(cid_s)
                    cx, cy, bw, bh = float(cx_s), float(cy_s), float(bw_s), float(bh_s)
                    x1 = max(0, int(round((cx - bw / 2.0) * w_img)))
                    y1 = max(0, int(round((cy - bh / 2.0) * h_img)))
                    x2 = min(w_img - 1, int(round((cx + bw / 2.0) * w_img)))
                    y2 = min(h_img - 1, int(round((cy + bh / 2.0) * h_img)))
                    cor = CORES_BGR[cid]
                    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
                    label_txt = ID_PARA_CLASSE[cid]
                    ty = max(14, y1 - 4)
                    cv2.putText(
                        img,
                        label_txt,
                        (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        cor,
                        1,
                        cv2.LINE_AA,
                    )
                out_path = pasta_sp / (Path(nome_img).stem + "_boxes.jpg")
                ok = cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise RuntimeError(f"Falha ao salvar amostra: {out_path}")
                total_geradas += 1
    return total_geradas


# =============================================================================
# VALIDAÇÕES OBRIGATÓRIAS
# =============================================================================

def validar_saida(
    por_dataset: Dict[str, Dict[str, List[RegistroImagem]]],
    metas: Dict[str, dict],
) -> dict:
    """Roda as cinco validações obrigatórias sobre o que foi gravado; qualquer falha interrompe o script."""
    resultados_val: dict = {}

    # 1. Conservação de objetos: saída = objetos de origem menos as caixas degeneradas descartadas.
    contas_conservacao = {}
    for dataset_id, splits_dict in por_dataset.items():
        origem_cls: Counter = Counter()
        desc_cls: Counter = Counter()
        for sp in SPLITS_SAIDA:
            for reg in splits_dict[sp]:
                origem_cls.update(reg.linhas_origem_por_classe)
                desc_cls.update(reg.descartadas_por_classe)

        saida_cls: Counter = Counter()
        for sp in SPLITS_SAIDA:
            lbl_dir = PASTA_SAIDA / "base" / dataset_id / sp / "labels"
            for lp in lbl_dir.glob("*.txt"):
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        saida_cls[int(ln.split()[0])] += 1

        detalhes_ds = {}
        for cls in metas[dataset_id]["classes_rotuladas"]:
            cid = CLASSE_PARA_ID[cls]
            orig_n = int(origem_cls[cid])
            desc_n = int(desc_cls[cid])
            out_n = int(saida_cls[cid])
            ok = out_n == (orig_n - desc_n)
            detalhes_ds[cls] = {
                "origem_mantidas": orig_n,
                "descartadas": desc_n,
                "saida": out_n,
                "ok": ok,
            }
            if not ok:
                raise AssertionError(
                    f"Falha de conservação em {dataset_id}/{cls}: {out_n} != {orig_n} - {desc_n}"
                )
        contas_conservacao[dataset_id] = detalhes_ds
    resultados_val["conservacao_objetos"] = {
        "passou": True,
        "detalhes": contas_conservacao,
    }

    # 2. Recalcula o dHash D4 sobre data_final/base: não pode haver par a até 4 bits entre divisões.
    pares_vazamento_por_ds = {}
    for dataset_id in por_dataset:
        itens_base = []
        for sp in SPLITS_SAIDA:
            img_dir = PASTA_SAIDA / "base" / dataset_id / sp / "images"
            for ip in sorted(img_dir.iterdir()):
                if ip.is_file():
                    h0, hd4 = dhash_d4_de_bytes(ip.read_bytes())
                    itens_base.append((sp, ip.name, h0, hd4))
        vazamentos = []
        for i in range(len(itens_base)):
            sp1, n1, h1, hd4_1 = itens_base[i]
            for j in range(i + 1, len(itens_base)):
                sp2, n2, h2, hd4_2 = itens_base[j]
                if sp1 == sp2:
                    continue
                dist = distancia_dhash_d4(h1, hd4_1, h2, hd4_2)
                if dist <= DHASH_MAX_HAMMING:
                    vazamentos.append((dist, sp1, n1, sp2, n2))
        pares_vazamento_por_ds[dataset_id] = len(vazamentos)
        if vazamentos:
            raise AssertionError(
                f"Vazamento dHash D4 detectado em {dataset_id}: {len(vazamentos)} pares (ex.: {vazamentos[0]})"
            )
    resultados_val["dhash_d4_zero_vazamento"] = {
        "passou": True,
        "pares_entre_splits_por_dataset": pares_vazamento_por_ds,
    }

    # 3. Cada imagem tem exatamente um .txt, em base/ e em yolo/.
    pares_verificados = {"base": 0, "yolo": 0}
    for grupo_raiz in ("base", "yolo"):
        raiz = PASTA_SAIDA / grupo_raiz
        for sub in sorted(raiz.iterdir()):
            if not sub.is_dir():
                continue
            for sp in SPLITS_SAIDA:
                img_dir = sub / sp / "images"
                lbl_dir = sub / sp / "labels"
                imgs = {p.stem for p in img_dir.iterdir() if p.is_file()}
                lbls = {p.stem for p in lbl_dir.glob("*.txt") if p.is_file()}
                if imgs != lbls:
                    raise AssertionError(
                        f"Descompasso imagem/label em {grupo_raiz}/{sub.name}/{sp}: "
                        f"sem_txt={len(imgs - lbls)}, sem_img={len(lbls - imgs)}"
                    )
                pares_verificados[grupo_raiz] += len(imgs)
    resultados_val["correspondencia_img_txt"] = {
        "passou": True,
        "pares_base": pares_verificados["base"],
        "pares_yolo": pares_verificados["yolo"],
    }

    # 4. Os data.yaml de yolo/ apontam para pastas que existem.
    yamls_ok = {}
    for cls in CLASSES_FINAIS:
        yp = PASTA_SAIDA / "yolo" / cls / "data.yaml"
        if not yp.is_file():
            raise AssertionError(f"data.yaml ausente: {yp}")
        cfg_y = yaml.safe_load(yp.read_text(encoding="utf-8"))
        base_y = (yp.parent / cfg_y.get("path", ".")).resolve()
        assert cfg_y["nc"] == 1 and cfg_y["names"] == [cls], f"YAML inválido: {yp}"
        for chave in ("train", "val", "test"):
            alvo = (base_y / cfg_y[chave]).resolve()
            if not alvo.is_dir():
                raise AssertionError(f"Caminho em {yp} ({chave}) não existe: {alvo}")
        yamls_ok[cls] = True
    resultados_val["yolo_data_yaml"] = {"passou": True, "classes": yamls_ok}

    # 5. A soma de contagens_gt.csv bate com os objetos de base/ por (dataset, split, classe).
    df_gt = pd.read_csv(PASTA_SAIDA / "contagens_gt.csv")
    soma_csv = (
        df_gt.groupby(["dataset", "split", "classe"])["contagem"].sum().to_dict()
    )
    soma_base: Dict[Tuple[str, str, str], int] = {}
    for dataset_id in por_dataset:
        classes_rot = metas[dataset_id]["classes_rotuladas"]
        for sp in SPLITS_SAIDA:
            c = Counter()
            lbl_dir = PASTA_SAIDA / "base" / dataset_id / sp / "labels"
            for lp in lbl_dir.glob("*.txt"):
                for ln in lp.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        c[ID_PARA_CLASSE[int(ln.split()[0])]] += 1
            for cls in classes_rot:
                soma_base[(dataset_id, sp, cls)] = int(c[cls])
    if soma_csv != soma_base:
        raise AssertionError(
            f"Soma de contagens_gt.csv diverge de base/: csv={soma_csv} vs base={soma_base}"
        )
    resultados_val["contagens_gt_csv"] = {
        "passou": True,
        "linhas_csv": int(len(df_gt)),
        "total_objetos_csv": int(df_gt["contagem"].sum()),
    }

    return resultados_val


def compactar_data_final() -> Tuple[Path, int, str]:
    """Compacta data_final/ em data_final.zip e devolve (caminho, tamanho em bytes, sha256)."""
    with zipfile.ZipFile(ARQUIVO_ZIP_SAIDA, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for arquivo in sorted(PASTA_SAIDA.rglob("*")):
            if arquivo.is_file():
                arcname = arquivo.relative_to(RAIZ_PROJETO).as_posix()
                zout.write(arquivo, arcname=arcname)
    tamanho = ARQUIVO_ZIP_SAIDA.stat().st_size
    sha = sha256_arquivo(ARQUIVO_ZIP_SAIDA)
    return ARQUIVO_ZIP_SAIDA, tamanho, sha


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main() -> None:
    """Executa a preparação de ponta a ponta e imprime o resumo por dataset e divisão."""
    print("=== FASE 1 — PREPARAÇÃO DA BASE FINAL ===")
    preparar_diretorios_saida()

    por_dataset: Dict[str, Dict[str, List[RegistroImagem]]] = {}
    metas: Dict[str, dict] = {}
    estatisticas_datasets: Dict[str, dict] = {}

    for cfg in DATASETS:
        print(f"\n[Dataset] {cfg.dataset_id} <- {cfg.caminho_zip.relative_to(RAIZ_PROJETO)}")
        registros, meta = carregar_dataset(cfg)
        metas[cfg.dataset_id] = meta

        grupos = agrupar_imagens(registros, cfg.dataset_id)
        tamanhos = sorted((len(g) for g in grupos), reverse=True)
        print(
            f"  Imagens válidas carregadas: {len(registros)} | "
            f"Excluídas: {len(meta['imagens_excluidas'])} | "
            f"Grupos: {len(grupos)} | Maior grupo: {tamanhos[0]} "
            f"({100.0 * tamanhos[0] / len(registros):.2f}%)"
        )
        if meta["txts_vazios_nao_excluidos"]:
            print(f"  AVISO (.txt vazios não excluídos): {meta['txts_vazios_nao_excluidos']}")

        splits_dict, grupos_por_split, omitidas_vt = dividir_grupos(
            grupos, meta["classes_rotuladas"], cfg.dataset_id
        )
        por_dataset[cfg.dataset_id] = splits_dict

        # Estatísticas deste dataset para o manifesto.
        splits_stats = {}
        descartadas_total = Counter()
        descartadas_mantidas = Counter()
        tipos_mantidas = Counter()
        for reg in registros:
            descartadas_total.update(reg.descartadas_por_classe)
        for sp in SPLITS_SAIDA:
            obj_sp = Counter()
            for reg in splits_dict[sp]:
                descartadas_mantidas.update(reg.descartadas_por_classe)
                tipos_mantidas.update(reg.tipos_origem)
                for cid, _ in reg.anotacoes:
                    obj_sp[ID_PARA_CLASSE[cid]] += 1
            splits_stats[sp] = {
                "grupos": grupos_por_split[sp],
                "imagens": len(splits_dict[sp]),
                "objetos_por_classe": {cls: int(obj_sp[cls]) for cls in meta["classes_rotuladas"]},
                "total_objetos": int(sum(obj_sp.values())),
            }
            print(
                f"  [{sp}] grupos={grupos_por_split[sp]} | imgs={len(splits_dict[sp])} | "
                f"objetos={splits_stats[sp]['objetos_por_classe']}"
            )

        estatisticas_datasets[cfg.dataset_id] = {
            **meta,
            "imagens_carregadas": len(registros),
            "total_grupos": len(grupos),
            "maior_grupo": tamanhos[0],
            "maior_grupo_pct": round(100.0 * tamanhos[0] / len(registros), 4),
            "imagens_omitidas_valid_test_dedup_grupo": omitidas_vt,
            "linhas_descartadas_degeneradas_todas": {
                ID_PARA_CLASSE[k]: int(v) for k, v in sorted(descartadas_total.items())
            },
            "linhas_descartadas_degeneradas_mantidas": {
                ID_PARA_CLASSE[k]: int(v) for k, v in sorted(descartadas_mantidas.items())
            },
            "tipos_origem_mantidas": dict(tipos_mantidas),
            "splits": splits_stats,
        }

    print("\n[Escrita] Gerando data_final/base/, data_final/yolo/ e contagens_gt.csv...")
    gravar_base_e_yolo(por_dataset, metas)

    print("[Amostras] Gerando data_final/_amostras/...")
    n_amostras = gerar_amostras_visuais(por_dataset)
    print(f"  Amostras geradas: {n_amostras}")

    print("[Validação] Executando validações obrigatórias...")
    validacoes = validar_saida(por_dataset, metas)
    validacoes["amostras_visuais"] = {"passou": n_amostras == len(DATASETS) * len(SPLITS_SAIDA) * AMOSTRAS_POR_SPLIT, "quantidade": n_amostras}

    # Estatísticas por detector YOLO.
    yolo_stats = {}
    for cls in CLASSES_FINAIS:
        ds_comp = [
            ds_id for ds_id, m in metas.items() if cls in m["classes_rotuladas"]
        ]
        sp_info = {}
        for sp in SPLITS_SAIDA:
            img_dir = PASTA_SAIDA / "yolo" / cls / sp / "images"
            lbl_dir = PASTA_SAIDA / "yolo" / cls / sp / "labels"
            n_imgs = sum(1 for p in img_dir.iterdir() if p.is_file())
            n_objs = 0
            n_vazios = 0
            for lp in lbl_dir.glob("*.txt"):
                linhas = [l for l in lp.read_text(encoding="utf-8").splitlines() if l.strip()]
                if not linhas:
                    n_vazios += 1
                n_objs += len(linhas)
            sp_info[sp] = {"imagens": n_imgs, "objetos": n_objs, "txts_vazios_negativos": n_vazios}
        yolo_stats[cls] = {"datasets": ds_comp, "splits": sp_info}

    manifesto = {
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "parametros": {
            "semente": SEMENTE,
            "proporcoes": {
                "train": PROPORCAO_TRAIN,
                "valid": PROPORCAO_VALID,
                "test": PROPORCAO_TEST,
            },
            "dhash_tamanho": DHASH_TAMANHO,
            "dhash_max_hamming": DHASH_MAX_HAMMING,
            "limite_max_fracao_grupo": LIMITE_MAX_FRACAO_GRUPO,
            "amostras_por_split": AMOSTRAS_POR_SPLIT,
            "classes_finais": CLASSE_PARA_ID,
        },
        "datasets": estatisticas_datasets,
        "detectores_yolo": yolo_stats,
        "validacoes": validacoes,
    }
    (PASTA_SAIDA / "manifesto.json").write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[Compactação] Gerando data_final.zip...")
    zip_path, zip_bytes, zip_sha = compactar_data_final()
    print(f"  Gerado: {zip_path.name} ({zip_bytes / 1024**2:.2f} MB | SHA-256: {zip_sha[:16]}...)")

    print("\n=== TODAS AS VALIDAÇÕES APROVADAS ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERRO / PARADA: {exc}", file=sys.stderr)
        sys.exit(1)
