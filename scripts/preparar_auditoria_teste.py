#!/usr/bin/env python3
"""Prepara a auditoria manual do GT da divisão test (Fase 2, parte C).

O que faz: desenha as caixas do GT sobre cada imagem única de test, com o nome da
classe em cada caixa e uma faixa no topo com a contagem por classe, para a revisão
manual. Cria também a planilha de auditoria e a lista de exclusões, ainda vazia.

Entradas: data_final/contagens_gt.csv e data_final/base/<dataset>/test/{images,labels}.
Saídas:
  - data_final/_auditoria/test/<dataset>/*_auditoria.jpg (a pasta é recriada a cada execução)
  - resultados/auditoria/exclusoes_teste.csv (só o cabeçalho)
  - resultados/auditoria/auditoria_teste.csv (uma linha por imagem de test)

Como rodar:
    .venv/bin/python scripts/preparar_auditoria_teste.py
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from typing import List

import cv2
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
PASTA_BASE = RAIZ_PROJETO / "data_final" / "base"
ARQUIVO_GT = RAIZ_PROJETO / "data_final" / "contagens_gt.csv"
PASTA_AUDITORIA_IMGS = RAIZ_PROJETO / "data_final" / "_auditoria" / "test"
PASTA_AUDITORIA_CSV = RAIZ_PROJETO / "resultados" / "auditoria"

ID_PARA_CLASSE = {0: "parafuso", 1: "porca", 2: "arruela"}
CORES_BGR = {
    0: (0, 140, 255),   # parafuso: laranja
    1: (255, 90, 0),    # porca: azul
    2: (0, 190, 0),     # arruela: verde
}


def desenhar_auditoria(caminho_img: Path, caminho_lbl: Path, classes_rotuladas: List[str]) -> cv2.typing.MatLike:
    """Desenha as caixas do label YOLO na imagem e devolve o resultado com a faixa de contagens por classe."""
    img = cv2.imread(str(caminho_img), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Falha ao ler imagem: {caminho_img}")
    h_img, w_img = img.shape[:2]

    contagem_cls: Counter = Counter()
    linhas = [ln.strip() for ln in caminho_lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]

    for ln in linhas:
        cid_s, cx_s, cy_s, bw_s, bh_s = ln.split()
        cid = int(cid_s)
        cx, cy, bw, bh = float(cx_s), float(cy_s), float(bw_s), float(bh_s)
        x1 = max(0, int(round((cx - bw / 2.0) * w_img)))
        y1 = max(0, int(round((cy - bh / 2.0) * h_img)))
        x2 = min(w_img - 1, int(round((cx + bw / 2.0) * w_img)))
        y2 = min(h_img - 1, int(round((cy + bh / 2.0) * h_img)))

        cor = CORES_BGR[cid]
        nome_cls = ID_PARA_CLASSE[cid]
        contagem_cls[nome_cls] += 1

        # Traço de 1 px para não cobrir objetos pequenos ou encostados.
        cv2.rectangle(img, (x1, y1), (x2, y2), cor, 1)
        ty = max(12, y1 - 3)
        cv2.putText(
            img,
            nome_cls,
            (x1, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            cor,
            1,
            cv2.LINE_AA,
        )

    # A faixa entra acima da imagem, para não cobrir nenhum pixel original.
    linhas_resumo = [f"{cls}: {contagem_cls[cls]}" for cls in classes_rotuladas]
    linhas_resumo.append(f"total: {sum(contagem_cls.values())}")
    texto_barra = "  |  ".join(linhas_resumo)

    altura_faixa = 28
    faixa = cv2.copyMakeBorder(img, altura_faixa, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))

    x_cursor = 8
    y_txt = 19
    for idx, item in enumerate(linhas_resumo):
        cls_nome = item.split(":")[0]
        if cls_nome == "parafuso":
            cor_txt = CORES_BGR[0]
        elif cls_nome == "porca":
            cor_txt = CORES_BGR[1]
        elif cls_nome == "arruela":
            cor_txt = CORES_BGR[2]
        else:
            cor_txt = (240, 240, 240)
        cv2.putText(
            faixa,
            item,
            (x_cursor, y_txt),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            cor_txt,
            1,
            cv2.LINE_AA,
        )
        (tw, _), _ = cv2.getTextSize(item, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        x_cursor += tw + 14
        if idx < len(linhas_resumo) - 1:
            cv2.putText(
                faixa,
                "|",
                (x_cursor - 9, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

    return faixa


def main() -> None:
    """Gera as imagens desenhadas e os dois CSVs da auditoria de test."""
    df_gt = pd.read_csv(ARQUIVO_GT)
    df_test = df_gt[df_gt["split"] == "test"].copy()

    if PASTA_AUDITORIA_IMGS.exists():
        shutil.rmtree(PASTA_AUDITORIA_IMGS)
    PASTA_AUDITORIA_IMGS.mkdir(parents=True, exist_ok=True)
    PASTA_AUDITORIA_CSV.mkdir(parents=True, exist_ok=True)

    registros_auditoria: List[dict] = []
    contagem_por_dataset: Counter = Counter()

    for dataset_id in sorted(df_test["dataset"].unique()):
        sub_ds = df_test[df_test["dataset"] == dataset_id]
        classes_rot = sorted(
            sub_ds["classe"].unique(),
            key=lambda c: {"parafuso": 0, "porca": 1, "arruela": 2}[c],
        )
        imagens_unicas = sorted(sub_ds["imagem"].unique())

        pasta_out_ds = PASTA_AUDITORIA_IMGS / dataset_id
        pasta_out_ds.mkdir(parents=True, exist_ok=True)

        img_dir = PASTA_BASE / dataset_id / "test" / "images"
        lbl_dir = PASTA_BASE / dataset_id / "test" / "labels"

        for nome_img in imagens_unicas:
            caminho_img = img_dir / nome_img
            caminho_lbl = lbl_dir / Path(nome_img).with_suffix(".txt").name
            if not caminho_img.is_file() or not caminho_lbl.is_file():
                raise FileNotFoundError(
                    f"Par imagem/label ausente em base/{dataset_id}/test: {nome_img}"
                )

            img_desenhada = desenhar_auditoria(caminho_img, caminho_lbl, classes_rot)
            out_img = pasta_out_ds / (Path(nome_img).stem + "_auditoria.jpg")
            ok = cv2.imwrite(str(out_img), img_desenhada, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError(f"Falha ao salvar imagem de auditoria: {out_img}")

            registros_auditoria.append(
                {
                    "dataset": dataset_id,
                    "imagem": nome_img,
                    "caminho_da_imagem_desenhada": out_img.relative_to(RAIZ_PROJETO).as_posix(),
                    "gt_completo": "",
                    "observacao": "",
                }
            )
            contagem_por_dataset[dataset_id] += 1

    # Lista de exclusões vazia (só o cabeçalho); o avaliador a lê com --exclusoes.
    df_exc_vazio = pd.DataFrame(columns=["dataset", "imagem", "motivo"])
    df_exc_vazio.to_csv(PASTA_AUDITORIA_CSV / "exclusoes_teste.csv", index=False, encoding="utf-8")

    # Planilha de auditoria: uma linha por imagem única de test.
    df_aud = pd.DataFrame(
        registros_auditoria,
        columns=[
            "dataset",
            "imagem",
            "caminho_da_imagem_desenhada",
            "gt_completo",
            "observacao",
        ],
    )
    df_aud.to_csv(PASTA_AUDITORIA_CSV / "auditoria_teste.csv", index=False, encoding="utf-8")

    print("=== PREPARAÇÃO DA AUDITORIA DO GT (SPLIT TEST) ===")
    for ds, qtd in sorted(contagem_por_dataset.items()):
        print(f"  {ds}: {qtd} imagens")
    print(f"  TOTAL: {sum(contagem_por_dataset.values())} imagens")
    print(f"Arquivos gerados em: {PASTA_AUDITORIA_IMGS} e {PASTA_AUDITORIA_CSV}")


if __name__ == "__main__":
    main()
