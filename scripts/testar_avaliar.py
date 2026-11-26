#!/usr/bin/env python3
"""Testes do avaliador scripts/avaliar.py (Fase 2, parte B).

O que faz: monta previsões sintéticas a partir do GT do test (GT como previsão, +1 em
todas as porcas, erro misto em nutsnbolts, linha faltando, linha extra, lista de
exclusões) e confere se MAE, viés, acerto exato e acerto ±1 saem com os valores
esperados. Ao final faz um teste de escrita de arquivos numa pasta temporária.

Entradas: data_final/contagens_gt.csv (lido pelo próprio avaliador).
Saídas: mensagens no terminal. Não grava nada no repositório.

Como rodar:
    .venv/bin/python scripts/testar_avaliar.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

# avaliar.py fica na mesma pasta e não é um pacote, então entra pelo sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import avaliar as av  # noqa: E402


def _obter_metrica(
    df_m: pd.DataFrame,
    metodo: str,
    escopo: str,
    classe: str,
    dataset: str,
    metrica: str,
) -> float:
    """Devolve o valor de uma métrica; falha se o filtro não casar com exatamente uma linha."""
    sub = df_m[
        (df_m["metodo"] == metodo)
        & (df_m["escopo"] == escopo)
        & (df_m["classe"] == classe)
        & (df_m["dataset"] == dataset)
        & (df_m["metrica"] == metrica)
    ]
    assert len(sub) == 1, f"Esperada 1 linha para {(metodo, escopo, classe, dataset, metrica)}, obtidas {len(sub)}"
    return float(sub["valor"].iloc[0])


def main() -> None:
    """Roda os 6 testes do avaliador mais o teste de escrita de arquivos."""
    df_gt_full = av.carregar_gt(av.GT_PADRAO)
    df_gt_test = df_gt_full[df_gt_full["split"] == "test"].copy().reset_index(drop=True)

    print("=== VALIDAÇÃO DO AVALIADOR (scripts/avaliar.py) ===\n")

    # -------------------------------------------------------------------------
    # Teste 1: GT usado como previsão -> erro = 0, acerto exato = 100%
    # -------------------------------------------------------------------------
    prev1 = df_gt_test.copy()
    prev1.insert(0, "metodo", "oraculo")
    por_img1, _ = av.validar_e_casar(df_gt_full, prev1, split="test")
    m1 = av.calcular_metricas(por_img1)

    for met in ("mae", "rmse", "vies", "mape_pct"):
        vals = m1[m1["metrica"] == met]["valor"]
        assert all(math.isclose(v, 0.0, abs_tol=1e-12) for v in vals), f"Teste 1 falhou em {met}"
    for met in ("acerto_exato_pct", "acerto_tol1_pct"):
        vals = m1[m1["metrica"] == met]["valor"]
        assert all(math.isclose(v, 100.0, abs_tol=1e-12) for v in vals), f"Teste 1 falhou em {met}"

    print(
        "[TESTE 1] PASSOU — GT como previsão:\n"
        f"  geral: n_imagens={int(_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'n_imagens'))}, "
        f"MAE={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'mae'):.4f}, "
        f"RMSE={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'rmse'):.4f}, "
        f"vies={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'vies'):.4f}, "
        f"MAPE={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'mape_pct'):.4f}% "
        f"(n_mape={int(_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'n_mape'))}), "
        f"exato={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'acerto_exato_pct'):.2f}%, "
        f"tol1={_obter_metrica(m1, 'oraculo', 'geral', 'TODAS', 'TODOS', 'acerto_tol1_pct'):.2f}%"
    )

    # -------------------------------------------------------------------------
    # Teste 2: +1 em todas as contagens da classe 'porca'
    # -------------------------------------------------------------------------
    prev2 = df_gt_test.copy()
    prev2.insert(0, "metodo", "mais1_porca")
    prev2.loc[prev2["classe"] == "porca", "contagem"] += 1
    por_img2, _ = av.validar_e_casar(df_gt_full, prev2, split="test")
    m2 = av.calcular_metricas(por_img2)

    assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", "porca", "TODOS", "mae"), 1.0)
    assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", "porca", "TODOS", "rmse"), 1.0)
    assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", "porca", "TODOS", "vies"), 1.0)
    assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", "porca", "TODOS", "acerto_exato_pct"), 0.0)
    assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", "porca", "TODOS", "acerto_tol1_pct"), 100.0)
    for cls_ok in ("parafuso", "arruela"):
        assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", cls_ok, "TODOS", "mae"), 0.0)
        assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", cls_ok, "TODOS", "vies"), 0.0)
        assert math.isclose(_obter_metrica(m2, "mais1_porca", "classe", cls_ok, "TODOS", "acerto_exato_pct"), 100.0)

    print(
        "[TESTE 2] PASSOU — +1 em todas as contagens de 'porca':\n"
        f"  porca: MAE={_obter_metrica(m2, 'mais1_porca', 'classe', 'porca', 'TODOS', 'mae'):.4f}, "
        f"RMSE={_obter_metrica(m2, 'mais1_porca', 'classe', 'porca', 'TODOS', 'rmse'):.4f}, "
        f"vies={_obter_metrica(m2, 'mais1_porca', 'classe', 'porca', 'TODOS', 'vies'):+.4f}, "
        f"exato={_obter_metrica(m2, 'mais1_porca', 'classe', 'porca', 'TODOS', 'acerto_exato_pct'):.2f}%, "
        f"tol1={_obter_metrica(m2, 'mais1_porca', 'classe', 'porca', 'TODOS', 'acerto_tol1_pct'):.2f}%\n"
        f"  parafuso: MAE={_obter_metrica(m2, 'mais1_porca', 'classe', 'parafuso', 'TODOS', 'mae'):.4f}, "
        f"exato={_obter_metrica(m2, 'mais1_porca', 'classe', 'parafuso', 'TODOS', 'acerto_exato_pct'):.2f}% | "
        f"arruela: MAE={_obter_metrica(m2, 'mais1_porca', 'classe', 'arruela', 'TODOS', 'mae'):.4f}, "
        f"exato={_obter_metrica(m2, 'mais1_porca', 'classe', 'arruela', 'TODOS', 'acerto_exato_pct'):.2f}%"
    )

    # -------------------------------------------------------------------------
    # Teste 3: erro misto (+2 em metade das imagens, -2 na outra metade de nutsnbolts).
    #          nutsnbolts/test tem 6 imagens (3 classes cada) e a menor contagem por classe
    #          é parafuso 15, porca 3 e arruela 5, então "contagem - 2" nunca fica negativa.
    # -------------------------------------------------------------------------
    prev3 = df_gt_test.copy()
    prev3.insert(0, "metodo", "misto_nutsnbolts")
    imgs_nb = sorted(prev3.loc[prev3["dataset"] == "nutsnbolts", "imagem"].unique())
    assert len(imgs_nb) == 6
    metade_mais = set(imgs_nb[:3])
    metade_menos = set(imgs_nb[3:])

    mask_mais = (prev3["dataset"] == "nutsnbolts") & (prev3["imagem"].isin(metade_mais))
    mask_menos = (prev3["dataset"] == "nutsnbolts") & (prev3["imagem"].isin(metade_menos))
    prev3.loc[mask_mais, "contagem"] += 2
    prev3.loc[mask_menos, "contagem"] -= 2

    por_img3, _ = av.validar_e_casar(df_gt_full, prev3, split="test")
    m3 = av.calcular_metricas(por_img3)

    for cls in ("parafuso", "porca", "arruela"):
        mae_c = _obter_metrica(m3, "misto_nutsnbolts", "classe_dataset", cls, "nutsnbolts", "mae")
        rmse_c = _obter_metrica(m3, "misto_nutsnbolts", "classe_dataset", cls, "nutsnbolts", "rmse")
        vies_c = _obter_metrica(m3, "misto_nutsnbolts", "classe_dataset", cls, "nutsnbolts", "vies")
        ex_c = _obter_metrica(m3, "misto_nutsnbolts", "classe_dataset", cls, "nutsnbolts", "acerto_exato_pct")
        t1_c = _obter_metrica(m3, "misto_nutsnbolts", "classe_dataset", cls, "nutsnbolts", "acerto_tol1_pct")
        assert math.isclose(mae_c, 2.0)
        assert math.isclose(rmse_c, 2.0)
        assert math.isclose(vies_c, 0.0)
        assert math.isclose(ex_c, 0.0)
        assert math.isclose(t1_c, 0.0)

    print(
        "[TESTE 3] PASSOU — Erro misto em nutsnbolts (3 imgs +2, 3 imgs -2 nas 3 classes):\n"
        "  Conta por classe em nutsnbolts: soma(erro) = 3*(+2) + 3*(-2) = 0 -> vies = 0/6 = 0.0000; "
        "soma(|erro|) = 6*2 = 12 -> MAE = 12/6 = 2.0000; RMSE = sqrt(6*4/6) = 2.0000; exato=0%, tol1=0%.\n"
        f"  Verificado em parafuso/porca/arruela: MAE=2.0000, RMSE=2.0000, vies=0.0000"
    )

    # -------------------------------------------------------------------------
    # Teste 4: Remover 1 linha das previsões -> erro de casamento
    # -------------------------------------------------------------------------
    prev4 = df_gt_test.iloc[1:].copy()
    prev4.insert(0, "metodo", "faltando_1")
    try:
        av.validar_e_casar(df_gt_full, prev4, split="test")
        raise AssertionError("Teste 4 deveria ter falhado com ErroValidacaoAvaliacao")
    except av.ErroValidacaoAvaliacao as exc:
        print(f"[TESTE 4] PASSOU — Linha removida gerou erro esperado:\n  {exc}")

    # -------------------------------------------------------------------------
    # Teste 5: Adicionar 1 linha inexistente no GT -> erro de casamento
    # -------------------------------------------------------------------------
    prev5 = df_gt_test.copy()
    prev5.insert(0, "metodo", "extra_1")
    linha_extra = pd.DataFrame(
        [
            {
                "metodo": "extra_1",
                "dataset": "washers_hardee",
                "split": "test",
                "imagem": prev5.loc[prev5["dataset"] == "washers_hardee", "imagem"].iloc[0],
                "classe": "parafuso",  # o GT de washers_hardee não tem a classe parafuso
                "contagem": 0,
            }
        ]
    )
    prev5 = pd.concat([prev5, linha_extra], ignore_index=True)
    try:
        av.validar_e_casar(df_gt_full, prev5, split="test")
        raise AssertionError("Teste 5 deveria ter falhado com ErroValidacaoAvaliacao")
    except av.ErroValidacaoAvaliacao as exc:
        print(f"[TESTE 5] PASSOU — Linha extra inexistente no GT gerou erro esperado:\n  {exc}")

    # -------------------------------------------------------------------------
    # Teste 6: Lista de exclusão com 2 imagens -> n_imagens cai exatamente 2
    # -------------------------------------------------------------------------
    imgs_wh = sorted(df_gt_test.loc[df_gt_test["dataset"] == "washers_hardee", "imagem"].unique())
    df_exc = pd.DataFrame(
        [
            {"dataset": "washers_hardee", "imagem": imgs_wh[0], "motivo": "teste_exclusao_1"},
            {"dataset": "washers_hardee", "imagem": imgs_wh[1], "motivo": "teste_exclusao_2"},
        ]
    )
    por_img6, info_exc6 = av.validar_e_casar(df_gt_full, prev1, split="test", df_exclusoes=df_exc)
    m6 = av.calcular_metricas(por_img6)

    n_geral_antes = int(_obter_metrica(m1, "oraculo", "geral", "TODAS", "TODOS", "n_imagens"))
    n_geral_depois = int(_obter_metrica(m6, "oraculo", "geral", "TODAS", "TODOS", "n_imagens"))
    n_arr_antes = int(_obter_metrica(m1, "oraculo", "classe", "arruela", "TODOS", "n_imagens"))
    n_arr_depois = int(_obter_metrica(m6, "oraculo", "classe", "arruela", "TODOS", "n_imagens"))
    n_wh_antes = int(_obter_metrica(m1, "oraculo", "classe_dataset", "arruela", "washers_hardee", "n_imagens"))
    n_wh_depois = int(_obter_metrica(m6, "oraculo", "classe_dataset", "arruela", "washers_hardee", "n_imagens"))
    n_par_antes = int(_obter_metrica(m1, "oraculo", "classe", "parafuso", "TODOS", "n_imagens"))
    n_par_depois = int(_obter_metrica(m6, "oraculo", "classe", "parafuso", "TODOS", "n_imagens"))

    assert info_exc6["imagens_excluidas_gt"] == 2
    assert n_geral_depois == n_geral_antes - 2
    assert n_arr_depois == n_arr_antes - 2
    assert n_wh_depois == n_wh_antes - 2
    assert n_par_depois == n_par_antes

    print(
        "[TESTE 6] PASSOU — Lista de exclusão com 2 imagens de washers_hardee:\n"
        f"  geral: {n_geral_antes} -> {n_geral_depois} (-2) | "
        f"classe arruela: {n_arr_antes} -> {n_arr_depois} (-2) | "
        f"washers_hardee/arruela: {n_wh_antes} -> {n_wh_depois} (-2) | "
        f"classe parafuso (não afetada): {n_par_antes} -> {n_par_depois} (0)"
    )

    # Teste extra: confere que avaliar() grava metricas.csv e por_imagem.csv (em pasta temporária).
    with tempfile.TemporaryDirectory(prefix="teste_avaliar_") as tmpdir:
        tmp = Path(tmpdir)
        prev_csv = tmp / "prev.csv"
        prev1.to_csv(prev_csv, index=False)
        m_cli, pi_cli, _ = av.avaliar(
            caminho_previsoes=prev_csv,
            caminho_gt=av.GT_PADRAO,
            nome_execucao="smoke_test",
            split="test",
            pasta_resultados=tmp / "out",
            imprimir=False,
        )
        assert (tmp / "out" / "smoke_test" / "metricas.csv").is_file()
        assert (tmp / "out" / "smoke_test" / "por_imagem.csv").is_file()
        assert len(m_cli) == len(m1) and len(pi_cli) == len(por_img1)

    print("\n=== TODOS OS 6 TESTES (E SMOKE TEST DE E/S) PASSARAM ===")


if __name__ == "__main__":
    main()
