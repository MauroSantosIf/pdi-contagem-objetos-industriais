#!/usr/bin/env python3
"""
Fase 2 — Avaliador de contagem de objetos (comum à pipeline clássica e YOLO).

Uso por linha de comando:
    .venv/bin/python scripts/avaliar.py \
        --previsoes caminho/para/previsoes.csv \
        --nome-execucao teste_classico \
        --split test \
        [--gt data_final/contagens_gt.csv] \
        [--exclusoes resultados/auditoria/exclusoes_teste.csv]
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RAIZ_PROJETO = Path(__file__).resolve().parent.parent
GT_PADRAO = RAIZ_PROJETO / "data_final" / "contagens_gt.csv"
PASTA_AVALIACAO_PADRAO = RAIZ_PROJETO / "resultados" / "avaliacao"

COLUNAS_GT = ("dataset", "split", "imagem", "classe", "contagem")
COLUNAS_PREVISOES = ("metodo", "dataset", "split", "imagem", "classe", "contagem")
COLUNAS_EXCLUSOES = ("dataset", "imagem", "motivo")
CHAVE_BASE = ["dataset", "split", "imagem", "classe"]


class ErroValidacaoAvaliacao(ValueError):
    """Erro de validação de entrada ou de casamento entre GT e previsões."""


def _validar_colunas(df: pd.DataFrame, obrigatorias: Tuple[str, ...], nome: str) -> None:
    faltantes = [c for c in obrigatorias if c not in df.columns]
    if faltantes:
        raise ErroValidacaoAvaliacao(
            f"[{nome}] Colunas obrigatórias ausentes: {faltantes}. "
            f"Encontradas: {list(df.columns)}"
        )


def validar_contagens(df: pd.DataFrame, nome: str) -> pd.DataFrame:
    """Garante que a coluna 'contagem' só possui inteiros >= 0 (sem NaN ou floats fracionários)."""
    out = df.copy()
    if out["contagem"].isna().any():
        linhas = out.index[out["contagem"].isna()].tolist()[:10]
        raise ErroValidacaoAvaliacao(
            f"[{nome}] Contagens ausentes (NaN) nas linhas (índice): {linhas}"
        )

    # Rejeita booleanos explicitamente (em Python/pandas bool é subclasse de int)
    if out["contagem"].map(lambda v: isinstance(v, (bool, np.bool_))).any():
        raise ErroValidacaoAvaliacao(f"[{nome}] Valores booleanos não são aceitos em 'contagem'.")

    try:
        numerico = pd.to_numeric(out["contagem"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ErroValidacaoAvaliacao(
            f"[{nome}] Contagens não numéricas encontradas em 'contagem': {exc}"
        ) from exc

    if not np.all(np.isfinite(numerico.to_numpy(dtype=float))):
        raise ErroValidacaoAvaliacao(f"[{nome}] Contagens não finitas (inf/-inf) encontradas.")

    valores_f = numerico.to_numpy(dtype=float)
    nao_inteiros = ~np.isclose(valores_f, np.round(valores_f), atol=0.0, rtol=0.0)
    if np.any(nao_inteiros):
        exemplos = out.loc[nao_inteiros, ["dataset", "imagem", "classe", "contagem"]].head(5).to_dict("records")
        raise ErroValidacaoAvaliacao(
            f"[{nome}] Contagens não inteiras encontradas ({int(nao_inteiros.sum())} linhas). "
            f"Exemplos: {exemplos}"
        )

    negativos = valores_f < 0
    if np.any(negativos):
        exemplos = out.loc[negativos, ["dataset", "imagem", "classe", "contagem"]].head(5).to_dict("records")
        raise ErroValidacaoAvaliacao(
            f"[{nome}] Contagens negativas encontradas ({int(negativos.sum())} linhas). "
            f"Exemplos: {exemplos}"
        )

    out["contagem"] = np.round(valores_f).astype(np.int64)
    return out


def carregar_gt(caminho_gt: Path) -> pd.DataFrame:
    if not caminho_gt.is_file():
        raise FileNotFoundError(f"Arquivo de GT não encontrado: {caminho_gt}")
    df = pd.read_csv(caminho_gt)
    _validar_colunas(df, COLUNAS_GT, "GT")
    return validar_contagens(df, "GT")


def carregar_previsoes(caminho_prev: Path) -> pd.DataFrame:
    if not caminho_prev.is_file():
        raise FileNotFoundError(f"Arquivo de previsões não encontrado: {caminho_prev}")
    df = pd.read_csv(caminho_prev)
    _validar_colunas(df, COLUNAS_PREVISOES, "Previsões")
    if df["metodo"].isna().any() or (df["metodo"].astype(str).str.strip() == "").any():
        raise ErroValidacaoAvaliacao("[Previsões] Coluna 'metodo' contém valores vazios/NaN.")
    return validar_contagens(df, "Previsões")


def carregar_exclusoes(caminho_exclusoes: Optional[Path]) -> Optional[pd.DataFrame]:
    if caminho_exclusoes is None:
        return None
    if not caminho_exclusoes.is_file():
        raise FileNotFoundError(f"Arquivo de exclusões não encontrado: {caminho_exclusoes}")
    df = pd.read_csv(caminho_exclusoes)
    _validar_colunas(df, COLUNAS_EXCLUSOES, "Exclusões")
    return df


def validar_e_casar(
    df_gt: pd.DataFrame,
    df_prev: pd.DataFrame,
    split: Optional[str] = "test",
    df_exclusoes: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Aplica filtro de split e exclusões, valida casamento estrito 1:1 por método
    e retorna (df_por_imagem, resumo_exclusoes).
    """
    _validar_colunas(df_gt, COLUNAS_GT, "GT")
    _validar_colunas(df_prev, COLUNAS_PREVISOES, "Previsões")
    gt = validar_contagens(df_gt, "GT")
    prev = validar_contagens(df_prev, "Previsões")

    if split is not None:
        gt = gt[gt["split"] == split].copy()
        prev = prev[prev["split"] == split].copy()

    if gt.empty:
        raise ErroValidacaoAvaliacao(f"GT ficou vazio após filtro split='{split}'.")
    if prev.empty:
        raise ErroValidacaoAvaliacao(f"Previsões ficaram vazias após filtro split='{split}'.")

    # Aplicar lista de exclusões (se fornecida)
    n_imgs_excluidas_gt = 0
    n_linhas_excluidas_gt = 0
    n_linhas_excluidas_prev = 0
    chaves_exc_usadas: List[Tuple[str, str]] = []

    if df_exclusoes is not None and not df_exclusoes.empty:
        _validar_colunas(df_exclusoes, COLUNAS_EXCLUSOES, "Exclusões")
        pares_exc = set(
            zip(df_exclusoes["dataset"].astype(str), df_exclusoes["imagem"].astype(str))
        )
        mask_gt_exc = [
            (d, i) in pares_exc
            for d, i in zip(gt["dataset"].astype(str), gt["imagem"].astype(str))
        ]
        mask_prev_exc = [
            (d, i) in pares_exc
            for d, i in zip(prev["dataset"].astype(str), prev["imagem"].astype(str))
        ]
        gt_exc = gt[mask_gt_exc]
        n_linhas_excluidas_gt = int(len(gt_exc))
        n_imgs_excluidas_gt = int(gt_exc[["dataset", "imagem"]].drop_duplicates().shape[0])
        chaves_exc_usadas = sorted(
            set(zip(gt_exc["dataset"].astype(str), gt_exc["imagem"].astype(str)))
        )
        n_linhas_excluidas_prev = int(sum(mask_prev_exc))

        gt = gt[[not m for m in mask_gt_exc]].copy()
        prev = prev[[not m for m in mask_prev_exc]].copy()

    if gt.empty:
        raise ErroValidacaoAvaliacao("GT ficou vazio após aplicar lista de exclusões.")

    # Checa duplicatas de chave no GT
    dup_gt = gt.duplicated(subset=CHAVE_BASE, keep=False)
    if dup_gt.any():
        ex = gt.loc[dup_gt, CHAVE_BASE].head(5).to_dict("records")
        raise ErroValidacaoAvaliacao(f"[GT] Chaves duplicadas encontradas: {ex}")

    chaves_gt = set(gt[CHAVE_BASE].itertuples(index=False, name=None))
    metodos = sorted(prev["metodo"].astype(str).unique().tolist())

    partes_casadas: List[pd.DataFrame] = []
    for metodo in metodos:
        sub = prev[prev["metodo"].astype(str) == metodo].copy()
        dup_sub = sub.duplicated(subset=CHAVE_BASE, keep=False)
        if dup_sub.any():
            ex = sub.loc[dup_sub, CHAVE_BASE].head(5).to_dict("records")
            raise ErroValidacaoAvaliacao(
                f"[Previsões/{metodo}] Chaves duplicadas encontradas: {ex}"
            )

        chaves_sub = set(sub[CHAVE_BASE].itertuples(index=False, name=None))
        faltantes = sorted(chaves_gt - chaves_sub)
        extras = sorted(chaves_sub - chaves_gt)

        if faltantes:
            amostra = [
                f"{d}/{s}/{img}/{cls}" for d, s, img, cls in faltantes[:10]
            ]
            raise ErroValidacaoAvaliacao(
                f"[Casamento/{metodo}] Faltam {len(faltantes)} linha(s) de previsão "
                f"presentes no GT: {amostra}"
            )
        if extras:
            amostra = [
                f"{d}/{s}/{img}/{cls}" for d, s, img, cls in extras[:10]
            ]
            raise ErroValidacaoAvaliacao(
                f"[Casamento/{metodo}] Existem {len(extras)} linha(s) de previsão "
                f"sem correspondente no GT: {amostra}"
            )

        casado = pd.merge(
            gt[CHAVE_BASE + ["contagem"]].rename(columns={"contagem": "contagem_gt"}),
            sub[CHAVE_BASE + ["contagem"]].rename(columns={"contagem": "contagem_prevista"}),
            on=CHAVE_BASE,
            how="inner",
            validate="one_to_one",
        )
        casado.insert(0, "metodo", metodo)
        partes_casadas.append(casado)

    por_imagem = pd.concat(partes_casadas, ignore_index=True)
    por_imagem["erro"] = por_imagem["contagem_prevista"] - por_imagem["contagem_gt"]
    por_imagem["erro_abs"] = por_imagem["erro"].abs()
    por_imagem["erro_quad"] = por_imagem["erro"] ** 2
    por_imagem["acerto_exato"] = (por_imagem["erro"] == 0).astype(int)
    por_imagem["acerto_tol1"] = (por_imagem["erro_abs"] <= 1).astype(int)
    por_imagem["ape_pct"] = np.where(
        por_imagem["contagem_gt"] > 0,
        (por_imagem["erro_abs"] / por_imagem["contagem_gt"]) * 100.0,
        np.nan,
    )
    por_imagem = por_imagem.sort_values(
        by=["erro_abs", "metodo", "dataset", "imagem", "classe"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)

    info_exclusoes = {
        "imagens_excluidas_gt": n_imgs_excluidas_gt,
        "linhas_excluidas_gt": n_linhas_excluidas_gt,
        "linhas_excluidas_previsoes": n_linhas_excluidas_prev,
        "chaves_excluidas": chaves_exc_usadas,
    }
    return por_imagem, info_exclusoes


def _calcular_dict_metricas(df_sub: pd.DataFrame) -> Dict[str, float]:
    n_imagens = int(df_sub[["dataset", "imagem"]].drop_duplicates().shape[0])
    n_avaliacoes = int(len(df_sub))
    soma_gt = int(df_sub["contagem_gt"].sum())
    soma_prev = int(df_sub["contagem_prevista"].sum())

    erro = df_sub["erro"].to_numpy(dtype=float)
    erro_abs = df_sub["erro_abs"].to_numpy(dtype=float)
    erro_quad = df_sub["erro_quad"].to_numpy(dtype=float)

    mae = float(np.mean(erro_abs))
    rmse = float(np.sqrt(np.mean(erro_quad)))
    vies = float(np.mean(erro))
    acerto_exato_pct = float(np.mean(df_sub["acerto_exato"].to_numpy(dtype=float)) * 100.0)
    acerto_tol1_pct = float(np.mean(df_sub["acerto_tol1"].to_numpy(dtype=float)) * 100.0)

    mask_pos = df_sub["contagem_gt"] > 0
    n_mape = int(mask_pos.sum())
    if n_mape > 0:
        mape_pct = float(df_sub.loc[mask_pos, "ape_pct"].mean())
    else:
        mape_pct = float("nan")

    return {
        "n_imagens": float(n_imagens),
        "n_avaliacoes": float(n_avaliacoes),
        "soma_gt": float(soma_gt),
        "soma_prevista": float(soma_prev),
        "mae": mae,
        "rmse": rmse,
        "vies": vies,
        "mape_pct": mape_pct,
        "n_mape": float(n_mape),
        "acerto_exato_pct": acerto_exato_pct,
        "acerto_tol1_pct": acerto_tol1_pct,
    }


def calcular_metricas(por_imagem: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas em formato longo (metodo, escopo, classe, dataset, metrica, valor)
    para os escopos:
      - geral (por metodo)
      - classe (por metodo x classe)
      - classe_dataset (por metodo x classe x dataset)
    """
    linhas: List[dict] = []

    def _adicionar(metodo: str, escopo: str, classe: str, dataset: str, sub: pd.DataFrame) -> None:
        m = _calcular_dict_metricas(sub)
        for nome_m, val in m.items():
            linhas.append(
                {
                    "metodo": metodo,
                    "escopo": escopo,
                    "classe": classe,
                    "dataset": dataset,
                    "metrica": nome_m,
                    "valor": val,
                }
            )

    for metodo, df_m in por_imagem.groupby("metodo", sort=True):
        _adicionar(str(metodo), "geral", "TODAS", "TODOS", df_m)
        for classe, df_c in df_m.groupby("classe", sort=True):
            _adicionar(str(metodo), "classe", str(classe), "TODOS", df_c)
            for dataset, df_cd in df_c.groupby("dataset", sort=True):
                _adicionar(str(metodo), "classe_dataset", str(classe), str(dataset), df_cd)

    return pd.DataFrame(
        linhas, columns=["metodo", "escopo", "classe", "dataset", "metrica", "valor"]
    )


def formatar_resumo_tabela(df_metricas: pd.DataFrame) -> str:
    """Converte o DataFrame longo de métricas em uma tabela resumida legível."""
    pivot = df_metricas.pivot(
        index=["metodo", "escopo", "classe", "dataset"],
        columns="metrica",
        values="valor",
    ).reset_index()

    ordem_escopo = {"geral": 0, "classe": 1, "classe_dataset": 2}
    pivot["_ord"] = pivot["escopo"].map(ordem_escopo)
    pivot = pivot.sort_values(["metodo", "_ord", "classe", "dataset"]).drop(columns=["_ord"])

    cols_exibir = [
        "metodo",
        "escopo",
        "classe",
        "dataset",
        "n_imagens",
        "soma_gt",
        "soma_prevista",
        "mae",
        "rmse",
        "vies",
        "mape_pct",
        "n_mape",
        "acerto_exato_pct",
        "acerto_tol1_pct",
    ]
    pivot = pivot[cols_exibir].copy()
    for c in ("n_imagens", "soma_gt", "soma_prevista", "n_mape"):
        pivot[c] = pivot[c].astype(int)
    for c in ("mae", "rmse", "vies", "mape_pct", "acerto_exato_pct", "acerto_tol1_pct"):
        pivot[c] = pivot[c].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    return pivot.to_string(index=False)


def avaliar(
    caminho_previsoes: Path,
    caminho_gt: Path = GT_PADRAO,
    nome_execucao: Optional[str] = None,
    split: Optional[str] = "test",
    caminho_exclusoes: Optional[Path] = None,
    pasta_resultados: Path = PASTA_AVALIACAO_PADRAO,
    imprimir: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Executa avaliação completa, salva metricas.csv e por_imagem.csv e imprime resumo."""
    df_gt = carregar_gt(caminho_gt)
    df_prev = carregar_previsoes(caminho_previsoes)
    df_exc = carregar_exclusoes(caminho_exclusoes)

    por_imagem, info_exc = validar_e_casar(df_gt, df_prev, split=split, df_exclusoes=df_exc)
    metricas = calcular_metricas(por_imagem)

    if not nome_execucao:
        nome_execucao = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pasta_saida = pasta_resultados / nome_execucao
    pasta_saida.mkdir(parents=True, exist_ok=True)

    metricas.to_csv(pasta_saida / "metricas.csv", index=False, encoding="utf-8")
    por_imagem.to_csv(pasta_saida / "por_imagem.csv", index=False, encoding="utf-8")

    if imprimir:
        print(f"=== AVALIAÇÃO DE CONTAGEM ({nome_execucao}) | split={split} ===")
        print(
            f"Exclusões aplicadas: {info_exc['imagens_excluidas_gt']} imagem(ns) "
            f"({info_exc['linhas_excluidas_gt']} linha(s) de GT, "
            f"{info_exc['linhas_excluidas_previsoes']} linha(s) de previsões)"
        )
        print()
        print(formatar_resumo_tabela(metricas))
        print(f"\nArquivos salvos em: {pasta_saida}")

    return metricas, por_imagem, info_exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avaliador de contagem de objetos industriais (clássico e YOLO)."
    )
    parser.add_argument(
        "--previsoes",
        type=Path,
        required=True,
        help="CSV de previsões (colunas: metodo, dataset, split, imagem, classe, contagem).",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=GT_PADRAO,
        help=f"CSV de Ground Truth (padrão: {GT_PADRAO}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split a avaliar (padrão: test).",
    )
    parser.add_argument(
        "--exclusoes",
        type=Path,
        default=None,
        help="CSV opcional de exclusões (colunas: dataset, imagem, motivo).",
    )
    parser.add_argument(
        "--nome-execucao",
        type=str,
        default=None,
        help="Nome da subpasta em resultados/avaliacao/<nome_execucao>/.",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=PASTA_AVALIACAO_PADRAO,
        help=f"Diretório base de saída (padrão: {PASTA_AVALIACAO_PADRAO}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    avaliar(
        caminho_previsoes=args.previsoes,
        caminho_gt=args.gt,
        nome_execucao=args.nome_execucao,
        split=args.split,
        caminho_exclusoes=args.exclusoes,
        pasta_resultados=args.saida,
        imprimir=True,
    )
