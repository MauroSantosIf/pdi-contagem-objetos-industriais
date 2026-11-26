#!/usr/bin/env python3
"""Detectores YOLO por classe (parafuso, porca, arruela), executados localmente (Fase 4A-LOCAL).

O que faz: reproduz as células 2 a 9 do notebook notebooks/etapa4_yolo_colab.ipynb com
caminhos locais, sem Google Drive e sem exigir GPU (usa a GPU se existir, senão a CPU).
Os padrões de treino daqui (yolo26n, 50 épocas, patience 10, batch 8) são mais leves que os
do notebook (yolo26s, 100 épocas, patience 20, batch 16). O resultado oficial do projeto é o
do Colab, em resultados/yolo/; as execuções locais parciais ficam em resultados/yolo_local_parcial/.
Treina um detector por classe, escolhe o limiar de confiança de cada uma no valid e,
depois de congelar a configuração, roda o test uma única vez.

Entradas: data_final/yolo/<classe>/ (imagens e labels), data_final/contagens_gt.csv e os
pesos pré-treinados (padrão yolo26n.pt, baixado para <saida>/_base/).
Saídas, em resultados/yolo/ (ou na pasta de --saida):
  - runs/<classe>/ (treino), calibracao/, deteccoes/, treino/, pesos/ e amostras_test/
  - config_yolo_final.json e config_yolo_final.sha256 (configuração congelada)
  - previsoes_valid.csv e previsoes_test.csv (formato lido por scripts/avaliar.py)
  - avaliacao/, manifesto_saidas.json, versoes_ambiente.json e log_<data>.txt

Subcomandos:
  treinar   treina um modelo por classe em <saida>/runs/<classe>/. Classes concluídas são
            puladas; --classe treina só uma; --retomar continua do last.pt.
  calibrar  escolhe o limiar de confiança de cada classe no valid e congela
            config_yolo_final.json (e o .sha256) antes de qualquer uso do test.
            Exige as 3 classes treinadas.
  testar    confere o hash do config e roda o test uma única vez (marcador
            TEST_EXECUCAO.json), grava previsoes_valid.csv e previsoes_test.csv, avalia
            com scripts/avaliar.py e salva os entregáveis. Exige as 3 classes treinadas.
  tudo      treinar, calibrar e testar em sequência.
  previa    compara a pipeline clássica com o detector YOLO no valid, para uma classe já
            treinada, em <saida>/previa/<classe>/. Não toca no test nem no config.

Regra do projeto: toda decisão usa apenas train e valid. O test só é lido em `testar`,
uma única vez, depois do congelamento, e nada pode ser ajustado depois de ver o resultado.

Como rodar (treino completo em segundo plano):
  setsid nohup .venv/bin/python scripts/yolo_local.py tudo > /dev/null 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import avaliar as av  # noqa: E402

DATA_FINAL = RAIZ / "data_final"
SAIDA_PADRAO = RAIZ / "resultados" / "yolo"

CLASSES = ["parafuso", "porca", "arruela"]
CONF_MIN_INFERENCIA = 0.05
LIMIARES = [round(0.05 * k, 2) for k in range(1, 19)]  # 0.05 ... 0.90
COLUNAS_PREVISOES = ["metodo", "dataset", "split", "imagem", "classe", "contagem"]
CORES_BGR = {"parafuso": (0, 140, 255), "porca": (255, 90, 0), "arruela": (0, 190, 0)}

# Valores esperados das Fases 1 e 2, conferidos antes de treinar.
DATA_FINAL_ZIP_SHA256 = "d542f8c06ea9bff1afe9ed25debaea46f11aeddee6661a9a8fabadcda77efc81"
AVALIAR_SHA256 = "61c65fb0d65b751b429b7203b64651b4b1f9b5c2fcf17c1e6979fe75986a8da5"
GT_LINHAS_ESPERADAS = 1280
GT_OBJETOS_ESPERADOS = 10980
IMAGENS_ESPERADAS = {
    "parafuso": {"train": 510, "valid": 40, "test": 40},
    "porca": {"train": 175, "valid": 15, "test": 15},
    "arruela": {"train": 361, "valid": 62, "test": 62},
}

# Pastas que ficam fora do manifesto de saídas (não são entregáveis).
EXCLUIR_DO_MANIFESTO = ("runs", "_dados", "_yaml", "_base", "previa")
ARQUIVO_CLASSICO_VALID = Path(__file__).resolve().parent.parent / "resultados" / "classico" / "previsoes_valid.csv"
CABECALHO_PREVIA = "PRÉVIA NO VALID — otimista para ambos os métodos; não é o resultado final"


# =============================================================================
# LOG (arquivo + console)
# =============================================================================

class Tee:
    """Duplica stdout/stderr para o arquivo de log. No arquivo, '\\r' descarta a linha parcial
    (barras de progresso viram uma linha por conclusão)."""

    def __init__(self, stream, arquivo):
        self._stream = stream
        self._arquivo = arquivo
        self._buf = ""

    def write(self, s):
        self._stream.write(s)
        for ch in s:
            if ch == "\r":
                self._buf = ""
            elif ch == "\n":
                self._arquivo.write(self._buf + "\n")
                self._buf = ""
            else:
                self._buf += ch
        self._arquivo.flush()
        return len(s)

    def flush(self):
        self._stream.flush()
        self._arquivo.flush()

    def __getattr__(self, nome):
        return getattr(self._stream, nome)


def iniciar_log(saida: Path) -> Path:
    """Cria log_<data>.txt em `saida` e passa a copiar stdout e stderr para ele."""
    saida.mkdir(parents=True, exist_ok=True)
    caminho = saida / f"log_{datetime.now():%Y%m%d_%H%M%S}.txt"
    arquivo = open(caminho, "a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.stdout, arquivo)
    sys.stderr = Tee(sys.stderr, arquivo)
    return caminho


def log(msg: str) -> None:
    """Imprime a mensagem com data e hora."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# =============================================================================
# UTILITÁRIOS
# =============================================================================

def sha256_arquivo(caminho, bloco: int = 1 << 20) -> str:
    """SHA-256 de um arquivo, lido em blocos de 1 MiB."""
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for parte in iter(lambda: f.read(bloco), b""):
            h.update(parte)
    return h.hexdigest()


def agora_utc() -> str:
    """Data e hora atuais em UTC, formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def dispositivo():
    """Índice da primeira GPU se o torch enxergar CUDA; senão "cpu"."""
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


class Contexto:
    """Caminhos e configuração única de uma execução."""

    def __init__(self, args: argparse.Namespace):
        self.saida = Path(args.saida).resolve()
        self.config = {
            "modelo": args.modelo,
            "imgsz": args.imgsz,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch": args.batch,
            "seed": args.seed,
            "deterministic": True,
            "max_det": args.max_det,
            "fraction": args.fraction,
        }
        self.metodo = Path(args.modelo).stem
        self.runs = self.saida / "runs"
        self.dados = self.saida / "_dados"
        self.yamls = self.saida / "_yaml"
        self.base = self.saida / "_base"
        self.calib = self.saida / "calibracao"
        self.deteccoes = self.saida / "deteccoes"
        self.config_path = self.saida / "config_yolo_final.json"
        self.config_sha_path = self.saida / "config_yolo_final.sha256"
        self.marcador_test = self.saida / "TEST_EXECUCAO.json"
        self.hash_esperado = getattr(args, "hash_esperado", None)
        self.retomar = bool(getattr(args, "retomar", False))
        self.classe = getattr(args, "classe", None)


def caminhos_run(ctx: Contexto, classe: str):
    """Devolve (pasta do run, best.pt, last.pt, train_concluido.json) da classe."""
    run = ctx.runs / classe
    return run, run / "weights" / "best.pt", run / "weights" / "last.pt", run / "train_concluido.json"


# =============================================================================
# DADOS E VERIFICAÇÕES
# =============================================================================

def verificar_dados() -> pd.DataFrame:
    """Confere avaliar.py, data_final/ (GT e contagens de imagens) e retorna o GT."""
    sha_av = sha256_arquivo(RAIZ / "scripts" / "avaliar.py")
    if sha_av != AVALIAR_SHA256:
        raise RuntimeError(f"scripts/avaliar.py DIVERGE do validado na Fase 2. PARANDO.\n  esperado: {AVALIAR_SHA256}\n  obtido  : {sha_av}")
    zip_local = RAIZ / "data_final.zip"
    if zip_local.is_file():
        sha_zip = sha256_arquivo(zip_local)
        if sha_zip != DATA_FINAL_ZIP_SHA256:
            raise RuntimeError(f"data_final.zip DIVERGE do hash da Fase 1. PARANDO.\n  esperado: {DATA_FINAL_ZIP_SHA256}\n  obtido  : {sha_zip}")
    df_gt = av.carregar_gt(DATA_FINAL / "contagens_gt.csv")
    if len(df_gt) != GT_LINHAS_ESPERADAS or int(df_gt["contagem"].sum()) != GT_OBJETOS_ESPERADOS:
        raise RuntimeError(
            f"GT inesperado: {len(df_gt)} linhas / {int(df_gt['contagem'].sum())} objetos "
            f"(esperado {GT_LINHAS_ESPERADAS} / {GT_OBJETOS_ESPERADOS})."
        )
    for classe, splits in IMAGENS_ESPERADAS.items():
        for split, n_esp in splits.items():
            n = len(list((DATA_FINAL / "yolo" / classe / split / "images").iterdir()))
            if n != n_esp:
                raise RuntimeError(f"{classe}/{split}: {n} imagens (esperado {n_esp}).")
    return df_gt


def registrar_versoes(ctx: Contexto) -> Dict[str, Any]:
    """Grava versoes_ambiente.json na primeira chamada; nas seguintes, apenas lê o arquivo."""
    caminho = ctx.saida / "versoes_ambiente.json"
    if caminho.is_file():
        return json.loads(caminho.read_text(encoding="utf-8"))
    import torch
    import ultralytics

    cpu = None
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.lower().startswith("model name"):
                cpu = ln.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    versoes = {
        "registrado_em_utc": agora_utc(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_torch": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "dispositivo": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": cpu,
        "threads_torch": torch.get_num_threads(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
    }
    ctx.saida.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(versoes, indent=2, ensure_ascii=False), encoding="utf-8")
    return versoes


def preparar_espelho_e_yaml(ctx: Contexto) -> Dict[str, Path]:
    """Espelha train/valid de data_final/yolo/<classe> em <saida>/_dados/<classe>/ (hardlinks; cópia se
    não for possível) para que o Ultralytics grave labels.cache fora de data_final/. Gera os data.yaml
    com caminho absoluto em <saida>/_yaml/. O test NÃO é espelhado nem declarado no yaml."""
    ctx.yamls.mkdir(parents=True, exist_ok=True)
    yamls: Dict[str, Path] = {}
    for classe in CLASSES:
        origem_raiz = DATA_FINAL / "yolo" / classe
        espelho = ctx.dados / classe
        for split in ("train", "valid"):
            for sub in ("images", "labels"):
                destino_dir = espelho / split / sub
                destino_dir.mkdir(parents=True, exist_ok=True)
                for arq in sorted((origem_raiz / split / sub).iterdir()):
                    destino = destino_dir / arq.name
                    if destino.exists():
                        continue
                    try:
                        os.link(arq, destino)
                    except OSError:
                        shutil.copy2(arq, destino)
        cfg_yaml = yaml.safe_load((origem_raiz / "data.yaml").read_text(encoding="utf-8"))
        nomes = list(cfg_yaml["names"].values()) if isinstance(cfg_yaml["names"], dict) else list(cfg_yaml["names"])
        if cfg_yaml["nc"] != 1 or nomes != [classe]:
            raise RuntimeError(f"data.yaml inesperado para {classe}: nc={cfg_yaml['nc']}, names={nomes}")
        cfg_yaml["path"] = str(espelho.resolve())
        cfg_yaml.pop("test", None)
        destino_yaml = ctx.yamls / f"{classe}.yaml"
        destino_yaml.write_text(yaml.safe_dump(cfg_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        yamls[classe] = destino_yaml
        for split in ("train", "valid"):
            n = len(list((espelho / split / "images").iterdir()))
            if n != IMAGENS_ESPERADAS[classe][split]:
                raise RuntimeError(f"Espelho {classe}/{split}: {n} imagens (esperado {IMAGENS_ESPERADAS[classe][split]}).")
    return yamls


def obter_pesos_base(ctx: Contexto) -> Path:
    """Baixa (uma vez) os pesos pré-treinados para <saida>/_base/ e confirma que carregam."""
    from ultralytics import YOLO

    alvo = ctx.base / ctx.config["modelo"]
    if not alvo.is_file():
        ctx.base.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(ctx.base)
        try:
            YOLO(ctx.config["modelo"])
        except Exception as erro:
            raise RuntimeError(
                f"NAO FOI POSSIVEL CARREGAR OS PESOS '{ctx.config['modelo']}'. PARANDO (nenhum modelo alternativo sera usado). "
                f"Erro original: {erro!r}"
            ) from erro
        finally:
            os.chdir(cwd)
    if not alvo.is_file():
        raise RuntimeError(f"Pesos {ctx.config['modelo']} não encontrados em {alvo} após o download.")
    try:
        modelo = YOLO(str(alvo))
    except Exception as erro:
        raise RuntimeError(f"Os pesos {alvo} não carregaram. PARANDO. Erro original: {erro!r}") from erro
    if modelo.task != "detect":
        raise RuntimeError(f"Os pesos {alvo} não são de detecção (task={modelo.task}).")
    return alvo


# =============================================================================
# TREINO (cada classe)
# =============================================================================

def _fmt(x) -> str:
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return "n/d"


def treinar_classe(ctx: Contexto, classe: str, yamls: Dict[str, Path], pesos_base: Path) -> None:
    """Treina (ou retoma) o detector de uma classe e grava train_concluido.json no fim; pula se já houver."""
    from ultralytics import YOLO

    run, best, last, marcador = caminhos_run(ctx, classe)
    if marcador.is_file():
        log(f"[{classe}] treino já concluído ({marcador.name}); pulando.")
        return
    t0 = time.time()

    def ao_fim_da_epoca(trainer):
        ep = trainer.epoch + 1
        if ep > trainer.epochs:  # o ultralytics dispara o callback mais uma vez na validação final do best.pt
            log(f"[{classe}] validação final do best.pt | mAP50={_fmt((trainer.metrics or {}).get('metrics/mAP50(B)'))}")
            return
        decorrido = time.time() - t0
        feitas = max(1, ep - trainer.start_epoch)
        eta_max = decorrido / feitas * max(0, trainer.epochs - ep)
        m = trainer.metrics or {}
        log(
            f"[{classe}] época {ep}/{trainer.epochs} | decorrido {decorrido / 60:.1f} min | ETA máx. {eta_max / 60:.1f} min | "
            f"fitness={_fmt(trainer.fitness)} | mAP50={_fmt(m.get('metrics/mAP50(B)'))} | mAP50-95={_fmt(m.get('metrics/mAP50-95(B)'))}"
        )

    if ctx.retomar and last.is_file():
        try:
            import torch

            ultima = int(torch.load(str(last), map_location="cpu", weights_only=False).get("epoch", -1)) + 1
            log(f"[{classe}] RETOMANDO a partir de {last}: última época salva = {ultima}; a próxima a rodar é a {ultima + 1}")
        except Exception as e:  # apenas informativo
            log(f"[{classe}] RETOMANDO a partir de {last} (não foi possível ler a época do checkpoint: {e})")
        modelo = YOLO(str(last))
        modelo.add_callback("on_fit_epoch_end", ao_fim_da_epoca)
        modelo.train(resume=True)
    else:
        if run.exists() and any(run.iterdir()):
            raise RuntimeError(
                f"[{classe}] {run} já existe e não está concluído. Use --retomar para retomar, "
                "ou apague a pasta manualmente se quiser recomeçar."
            )
        log(f"[{classe}] iniciando treino ({ctx.config['modelo']}, épocas={ctx.config['epochs']}, patience={ctx.config['patience']}, batch={ctx.config['batch']})")
        modelo = YOLO(str(pesos_base))
        modelo.add_callback("on_fit_epoch_end", ao_fim_da_epoca)
        modelo.train(
            data=str(yamls[classe]),
            imgsz=ctx.config["imgsz"],
            epochs=ctx.config["epochs"],
            patience=ctx.config["patience"],
            batch=ctx.config["batch"],
            seed=ctx.config["seed"],
            deterministic=ctx.config["deterministic"],
            max_det=ctx.config["max_det"],
            fraction=ctx.config["fraction"],
            project=str(ctx.runs),
            name=classe,
            exist_ok=True,
            device=dispositivo(),
        )
    if not best.is_file():
        raise RuntimeError(f"[{classe}] best.pt não encontrado em {best} após o treino.")
    marcador.write_text(
        json.dumps({"classe": classe, "concluido_em_utc": agora_utc(), "segundos_desta_execucao": round(time.time() - t0, 1)}, indent=2),
        encoding="utf-8",
    )
    log(f"[{classe}] concluído em {(time.time() - t0) / 60:.1f} min (esta execução). best.pt: {best}")


def classes_sem_treino_concluido(ctx: Contexto) -> List[str]:
    """Classes que não têm train_concluido.json (ou best.pt)."""
    faltam = []
    for classe in CLASSES:
        _, best, _, marcador = caminhos_run(ctx, classe)
        if not marcador.is_file() or not best.is_file():
            faltam.append(classe)
    return faltam


def exigir_treinos_concluidos(ctx: Contexto, comando: str) -> None:
    """Interrompe o comando se alguma das 3 classes não tiver terminado o treino, listando as que faltam."""
    faltam = classes_sem_treino_concluido(ctx)
    if faltam:
        raise RuntimeError(
            f"`{comando}` exige as {len(CLASSES)} classes treinadas. Sem train_concluido.json: {', '.join(faltam)}. "
            "Execute `treinar --classe <classe>` (ou `treinar`) para as que faltam."
        )


def cmd_treinar(ctx: Contexto) -> None:
    """Subcomando treinar: prepara espelho, yamls e pesos base e treina as classes pedidas."""
    classes = [ctx.classe] if getattr(ctx, "classe", None) else list(CLASSES)
    verificar_dados()
    versoes = registrar_versoes(ctx)
    log(f"Ambiente: {json.dumps(versoes, ensure_ascii=False)}")
    yamls = preparar_espelho_e_yaml(ctx)
    pesos_base = obter_pesos_base(ctx)
    log(f"Config: {json.dumps(ctx.config)}")
    log(f"Classes deste comando: {', '.join(classes)}")
    for i, classe in enumerate(classes, 1):
        log(f"=== TREINO {i}/{len(classes)}: {classe} ===")
        treinar_classe(ctx, classe, yamls, pesos_base)
    faltam = classes_sem_treino_concluido(ctx)
    log("Classes ainda sem treino concluído: " + (", ".join(faltam) if faltam else "nenhuma (pronto para calibrar + testar)"))


# =============================================================================
# CONTAGEM POR LIMIAR E INFERÊNCIA (mesma lógica do notebook)
# =============================================================================

def mascara_acima(confs, limiar):
    """Caixas com confiança >= limiar (tolerância de 1e-9 apenas contra ruído de ponto flutuante)."""
    return np.asarray(confs, dtype=np.float64) >= (float(limiar) - 1e-9)


def contar_acima(confs, limiar) -> int:
    """Contagem = número de caixas com confiança >= limiar."""
    return int(mascara_acima(confs, limiar).sum())


def linhas_gt_da_classe(df_gt: pd.DataFrame, classe: str, split: str) -> pd.DataFrame:
    """Linhas do GT de uma classe e divisão, em ordem estável (dataset, imagem)."""
    sub = df_gt[(df_gt["classe"] == classe) & (df_gt["split"] == split)]
    return sub.sort_values(["dataset", "imagem"]).reset_index(drop=True)


def inferir_deteccoes(model, classe, split, gt_cls, raiz_yolo, config) -> Dict[str, Any]:
    """Uma inferência com conf baixo; devolve {imagem: {dataset, conf, xyxy}}."""
    nomes = list(gt_cls["imagem"])
    pasta = Path(raiz_yolo) / classe / split / "images"
    caminhos = [str(pasta / n) for n in nomes]
    ausentes = [c for c in caminhos if not Path(c).is_file()]
    if ausentes:
        raise FileNotFoundError(f"{len(ausentes)} imagem(ns) do GT ausentes em {pasta}: {ausentes[:3]}")
    dataset_de = dict(zip(gt_cls["imagem"], gt_cls["dataset"]))
    det: Dict[str, Any] = {}
    for r in model.predict(
        source=caminhos,
        conf=CONF_MIN_INFERENCIA,
        imgsz=config["imgsz"],
        max_det=config["max_det"],
        batch=config["batch"],
        device=dispositivo(),
        half=False,
        verbose=False,
        stream=True,
    ):
        nome = Path(r.path).name
        det[nome] = {
            "dataset": dataset_de[nome],
            "conf": [float(x) for x in r.boxes.conf.cpu().numpy().tolist()],
            "xyxy": [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy().tolist()],
        }
    if set(det) != set(nomes):
        raise RuntimeError(f"Inferência incompleta em {classe}/{split}: {len(det)} de {len(nomes)} imagens.")
    return det


def salvar_deteccoes(det, caminho) -> None:
    """Grava as detecções (confiança e caixa por imagem) em JSON."""
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    Path(caminho).write_text(json.dumps(det), encoding="utf-8")


def carregar_deteccoes(caminho) -> Dict[str, Any]:
    """Lê o JSON gravado por salvar_deteccoes."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def montar_previsoes(gt_cls, det, limiar, metodo) -> pd.DataFrame:
    """Linhas no formato do avaliador, exatamente uma por linha do GT da classe."""
    linhas = []
    for r in gt_cls.itertuples(index=False):
        linhas.append(
            {
                "metodo": metodo,
                "dataset": r.dataset,
                "split": r.split,
                "imagem": r.imagem,
                "classe": r.classe,
                "contagem": contar_acima(det[r.imagem]["conf"], limiar),
            }
        )
    return pd.DataFrame(linhas, columns=COLUNAS_PREVISOES)


def gerar_previsoes(df_gt, split, deteccoes_por_classe, limiares_por_classe, metodo) -> pd.DataFrame:
    """Concatena as previsões das três classes, só com linhas que existem no GT da divisão."""
    partes = []
    for classe in CLASSES:
        gt_c = linhas_gt_da_classe(df_gt, classe, split)
        partes.append(montar_previsoes(gt_c, deteccoes_por_classe[classe], limiares_por_classe[classe], metodo))
    df = pd.concat(partes, ignore_index=True)
    return df.sort_values(["dataset", "imagem", "classe"]).reset_index(drop=True)


def calibrar_limiar(gt_cls, det, classe, pasta_tmp, metodo):
    """Varre os limiares e avalia cada um com avaliar.py (--split valid). Devolve (curva, limiar escolhido).

    Critério de escolha: menor MAE; em empate, menor |viés|; depois, menor limiar.
    """
    pasta_tmp = Path(pasta_tmp)
    pasta_tmp.mkdir(parents=True, exist_ok=True)
    caminho_gt = pasta_tmp / f"gt_valid_{classe}.csv"
    gt_cls.to_csv(caminho_gt, index=False, encoding="utf-8")
    linhas = []
    for lim in LIMIARES:
        prev = montar_previsoes(gt_cls, det, lim, metodo)
        caminho_prev = pasta_tmp / f"prev_valid_{classe}_{lim:.2f}.csv"
        prev.to_csv(caminho_prev, index=False, encoding="utf-8")
        metricas, _, _ = av.avaliar(
            caminho_previsoes=caminho_prev,
            caminho_gt=caminho_gt,
            nome_execucao=f"calib_{classe}_{lim:.2f}",
            split="valid",
            pasta_resultados=pasta_tmp / "avaliacao",
            imprimir=False,
        )
        m = metricas[(metricas["escopo"] == "classe") & (metricas["classe"] == classe)].set_index("metrica")["valor"]
        linhas.append(
            {
                "classe": classe,
                "limiar": lim,
                "mae": float(m["mae"]),
                "rmse": float(m["rmse"]),
                "vies": float(m["vies"]),
                "acerto_exato_pct": float(m["acerto_exato_pct"]),
                "acerto_tol1_pct": float(m["acerto_tol1_pct"]),
                "soma_gt": int(m["soma_gt"]),
                "soma_prevista": int(m["soma_prevista"]),
            }
        )
    curva = pd.DataFrame(linhas)
    ordenada = curva.assign(_mae=curva["mae"].round(10), _abs_vies=curva["vies"].abs().round(10)).sort_values(
        ["_mae", "_abs_vies", "limiar"]
    )
    return curva, float(ordenada.iloc[0]["limiar"])


# =============================================================================
# CALIBRAÇÃO E CONGELAMENTO (somente valid)
# =============================================================================

def cmd_calibrar(ctx: Contexto) -> None:
    """Subcomando calibrar: infere no valid, escolhe o limiar de cada classe e congela o config."""
    from ultralytics import YOLO

    if ctx.marcador_test.is_file() and json.loads(ctx.marcador_test.read_text(encoding="utf-8")).get("status") == "concluido":
        raise RuntimeError("O split test já foi executado e concluído: a calibração/config não pode mais ser regerada.")
    exigir_treinos_concluidos(ctx, "calibrar")
    df_gt = verificar_dados()
    versoes = registrar_versoes(ctx)
    ctx.calib.mkdir(parents=True, exist_ok=True)
    ctx.deteccoes.mkdir(parents=True, exist_ok=True)

    curvas, escolhidos = [], {}
    with tempfile.TemporaryDirectory(prefix="calib_yolo_") as tmp:
        for classe in CLASSES:
            _, best, _, marcador = caminhos_run(ctx, classe)
            if not marcador.is_file() or not best.is_file():
                raise RuntimeError(f"[{classe}] treino não concluído (execute `treinar` antes).")
            log(f"[{classe}] calibrando limiar no valid")
            gt_val = linhas_gt_da_classe(df_gt, classe, "valid")
            det_val = inferir_deteccoes(YOLO(str(best)), classe, "valid", gt_val, DATA_FINAL / "yolo", ctx.config)
            salvar_deteccoes(det_val, ctx.deteccoes / f"deteccoes_valid_{classe}.json")
            curva, limiar = calibrar_limiar(gt_val, det_val, classe, tmp, ctx.metodo)
            curva.to_csv(ctx.calib / f"curva_calibracao_{classe}.csv", index=False, encoding="utf-8")
            curvas.append(curva)
            mae_escolhido = float(curva.loc[curva["limiar"] == limiar, "mae"].iloc[0])
            escolhidos[classe] = {"limiar_conf": limiar, "mae_valid": mae_escolhido}
            log(f"[{classe}] limiar escolhido = {limiar:.2f} | MAE valid = {mae_escolhido:.3f}")

    curva_total = pd.concat(curvas, ignore_index=True)
    curva_total.to_csv(ctx.calib / "curva_calibracao.csv", index=False, encoding="utf-8")
    (ctx.calib / "limiares_escolhidos.json").write_text(json.dumps(escolhidos, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, len(CLASSES), figsize=(5 * len(CLASSES), 3.8))
    for ax, classe in zip(np.atleast_1d(eixos), CLASSES):
        c = curva_total[curva_total["classe"] == classe]
        ax.plot(c["limiar"], c["mae"], marker="o")
        ax.axvline(escolhidos[classe]["limiar_conf"], color="red", linestyle="--", label=f"escolhido = {escolhidos[classe]['limiar_conf']:.2f}")
        ax.set_title(f"{classe} (valid)")
        ax.set_xlabel("limiar de confiança")
        ax.set_ylabel("MAE")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ctx.calib / "curva_calibracao.png", dpi=150)
    plt.close(fig)

    congelar_config(ctx, escolhidos, versoes)


def congelar_config(ctx: Contexto, limiares: Dict[str, Any], versoes: Dict[str, Any]) -> None:
    """Grava config_yolo_final.json e o SHA-256 dele ANTES de qualquer uso do test."""
    pesos_base = ctx.base / ctx.config["modelo"]
    config_final = {
        "metodo": ctx.metodo,
        "congelado_em_utc": agora_utc(),
        "nota": "Congelado apos treino e calibracao em train/valid. O split test nao foi usado ate este ponto.",
        "modelo": ctx.config["modelo"],
        "modelo_pesos_sha256": sha256_arquivo(pesos_base) if pesos_base.is_file() else None,
        "config_treino": ctx.config,
        "calibracao": {
            "conf_minimo_inferencia": CONF_MIN_INFERENCIA,
            "grade_limiares": LIMIARES,
            "criterio": "menor MAE no valid (avaliar.py, split=valid); empate: menor |vies|, depois menor limiar",
            "regra_contagem": "numero de caixas com confianca >= limiar da classe",
        },
        "versoes": versoes,
        "data_final_zip_sha256": DATA_FINAL_ZIP_SHA256,
        "avaliar_py_sha256": AVALIAR_SHA256,
        "classes": {},
    }
    for classe in CLASSES:
        _, best, _, _ = caminhos_run(ctx, classe)
        config_final["classes"][classe] = {
            "best_pt": str(best.relative_to(ctx.saida)),
            "best_pt_sha256": sha256_arquivo(best),
            "limiar_conf": limiares[classe]["limiar_conf"],
            "mae_valid_no_limiar": limiares[classe]["mae_valid"],
        }
    ctx.config_path.write_text(json.dumps(config_final, indent=2, ensure_ascii=False), encoding="utf-8")
    hash_config = sha256_arquivo(ctx.config_path)
    ctx.config_sha_path.write_text(f"{hash_config}  config_yolo_final.json\n", encoding="utf-8")
    log(f"[CONGELADO] {ctx.config_path}")
    log(f"[SHA-256 config_yolo_final.json] {hash_config}")
    log("Limiares: " + json.dumps({c: v["limiar_conf"] for c, v in config_final["classes"].items()}))


# =============================================================================
# TEST (EXECUÇÃO ÚNICA) E ENTREGÁVEIS
# =============================================================================

def desenhar_amostra(caminho_img, classes_ds, deteccoes, limiares, gt_contagens, destino) -> Dict[str, int]:
    """Desenha as caixas com confiança >= limiar de cada classe do dataset, com a faixa de texto acima da imagem."""
    img = cv2.imread(str(caminho_img), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Imagem ilegível: {caminho_img}")
    nome = Path(caminho_img).name
    previstos = {}
    for classe in classes_ds:
        d = deteccoes[classe][nome]
        confs = np.asarray(d["conf"], dtype=np.float64)
        caixas = np.asarray(d["xyxy"], dtype=np.float64).reshape(-1, 4)
        manter = mascara_acima(confs, limiares[classe]) if len(confs) else np.zeros(0, dtype=bool)
        previstos[classe] = int(manter.sum())
        for (x1, y1, x2, y2), c in zip(caixas[manter], confs[manter]):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CORES_BGR[classe], 2)
            cv2.putText(img, f"{c:.2f}", (int(x1), max(10, int(y1) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CORES_BGR[classe], 1, cv2.LINE_AA)
    faixa = np.full((28, img.shape[1], 3), 245, dtype=np.uint8)
    x = 6
    for classe in classes_ds:
        txt = f"{classe}: prev={previstos[classe]} gt={gt_contagens[classe]}"
        cv2.putText(faixa, txt, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, CORES_BGR[classe], 1, cv2.LINE_AA)
        x += 9 * len(txt) + 14
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destino), np.vstack([faixa, img]))
    return previstos


def sortear_imagens(df_gt, dataset, split, n=3, seed=42) -> List[str]:
    """Sorteia, com semente fixa, até n imagens do dataset e da divisão para as amostras desenhadas."""
    nomes = sorted(df_gt[(df_gt["dataset"] == dataset) & (df_gt["split"] == split)]["imagem"].unique())
    return sorted(random.Random(seed).sample(nomes, min(n, len(nomes))))


def cmd_testar(ctx: Contexto) -> None:
    """Subcomando testar: roda o test uma única vez com o config congelado e salva os entregáveis."""
    from ultralytics import YOLO

    df_gt = verificar_dados()

    # --- Trava: o test só pode ser concluído uma vez ---
    if ctx.marcador_test.is_file():
        estado = json.loads(ctx.marcador_test.read_text(encoding="utf-8"))
        if estado.get("status") == "concluido":
            raise RuntimeError("O split test JÁ foi executado e concluído. Não execute de novo (regra: uma única vez).")
        log("AVISO: execução anterior iniciada e não concluída (nenhum resultado foi registrado); repetindo com o MESMO config.")

    exigir_treinos_concluidos(ctx, "testar")

    # --- 1. Conferência do SHA-256 do config congelado ---
    if not ctx.config_path.is_file() or not ctx.config_sha_path.is_file():
        raise RuntimeError("config_yolo_final.json/.sha256 não encontrados: execute `calibrar` antes.")
    hash_atual = sha256_arquivo(ctx.config_path)
    hash_registrado = ctx.config_sha_path.read_text(encoding="utf-8").split()[0]
    if hash_atual != hash_registrado:
        raise RuntimeError(f"SHA-256 do config DIVERGE do registrado. PARANDO.\n  registrado: {hash_registrado}\n  atual     : {hash_atual}")
    if ctx.hash_esperado and hash_atual != ctx.hash_esperado:
        raise RuntimeError(f"SHA-256 do config DIVERGE de --hash-esperado. PARANDO.\n  esperado: {ctx.hash_esperado}\n  atual   : {hash_atual}")
    log(f"SHA-256 do config conferido: {hash_atual}")

    cfg = json.loads(ctx.config_path.read_text(encoding="utf-8"))
    config_inf = cfg["config_treino"]  # parâmetros de inferência vêm do config congelado
    metodo = cfg["metodo"]
    limiares_final = {c: float(cfg["classes"][c]["limiar_conf"]) for c in CLASSES}
    ctx.marcador_test.write_text(
        json.dumps({"status": "iniciado", "config_sha256": hash_atual, "iniciado_em_utc": agora_utc()}, indent=2), encoding="utf-8"
    )

    # --- 2. Previsões do valid (detecções salvas na calibração e limiares congelados) ---
    det_valid = {c: carregar_deteccoes(ctx.deteccoes / f"deteccoes_valid_{c}.json") for c in CLASSES}
    prev_valid = gerar_previsoes(df_gt, "valid", det_valid, limiares_final, metodo)
    caminho_prev_valid = ctx.saida / "previsoes_valid.csv"
    prev_valid.to_csv(caminho_prev_valid, index=False, encoding="utf-8")

    # --- 3. Inferência no test (uma única vez) ---
    det_test: Dict[str, Any] = {}
    for classe in CLASSES:
        info = cfg["classes"][classe]
        best = ctx.saida / info["best_pt"]
        if sha256_arquivo(best) != info["best_pt_sha256"]:
            raise RuntimeError(f"best.pt de {classe} mudou desde o congelamento. PARANDO.")
        gt_test = linhas_gt_da_classe(df_gt, classe, "test")
        det_test[classe] = inferir_deteccoes(YOLO(str(best)), classe, "test", gt_test, DATA_FINAL / "yolo", config_inf)
        salvar_deteccoes(det_test[classe], ctx.deteccoes / f"deteccoes_test_{classe}.json")
        log(f"[{classe}] test: {len(det_test[classe])} imagens | limiar = {limiares_final[classe]:.2f}")

    prev_test = gerar_previsoes(df_gt, "test", det_test, limiares_final, metodo)
    caminho_prev_test = ctx.saida / "previsoes_test.csv"
    prev_test.to_csv(caminho_prev_test, index=False, encoding="utf-8")

    # --- 4. Avaliação com scripts/avaliar.py (casamento estrito com o GT) ---
    av_dir = ctx.saida / "avaliacao"
    gt_path = DATA_FINAL / "contagens_gt.csv"
    print("\n=== VALID (limiares congelados) ===")
    av.avaliar(caminho_prev_valid, gt_path, nome_execucao=f"{metodo}_valid", split="valid", pasta_resultados=av_dir)
    print("\n=== TEST (resultado final) ===")
    av.avaliar(caminho_prev_test, gt_path, nome_execucao=f"{metodo}_test", split="test", pasta_resultados=av_dir)

    ctx.marcador_test.write_text(
        json.dumps({"status": "concluido", "config_sha256": hash_atual, "concluido_em_utc": agora_utc()}, indent=2), encoding="utf-8"
    )
    log("TEST concluído (marcador TEST_EXECUCAO.json = concluido).")

    salvar_entregaveis(ctx, df_gt, det_test, limiares_final, hash_atual)


def salvar_entregaveis(ctx: Contexto, df_gt, det_test, limiares_final, hash_config) -> None:
    """Grava amostras_test/, treino/, pesos/ e o manifesto_saidas.json com o SHA-256 de cada arquivo."""
    # --- Imagens de test com as caixas previstas (3 por dataset, semente 42) ---
    amostras_dir = ctx.saida / "amostras_test"
    classes_do_dataset = df_gt.groupby("dataset")["classe"].unique().to_dict()
    for dataset, classes_ds in sorted(classes_do_dataset.items()):
        classes_ds = [c for c in CLASSES if c in set(classes_ds)]
        for nome in sortear_imagens(df_gt, dataset, "test", n=3, seed=42):
            gt_c = {
                c: int(
                    df_gt[(df_gt["dataset"] == dataset) & (df_gt["split"] == "test") & (df_gt["imagem"] == nome) & (df_gt["classe"] == c)][
                        "contagem"
                    ].iloc[0]
                )
                for c in classes_ds
            }
            origem = DATA_FINAL / "base" / dataset / "test" / "images" / nome
            desenhar_amostra(origem, classes_ds, det_test, limiares_final, gt_c, amostras_dir / dataset / (Path(nome).stem + "_yolo.jpg"))
    log(f"Amostras salvas em {amostras_dir}")

    # --- Artefatos de treino e pesos ---
    for classe in CLASSES:
        run, best, _, _ = caminhos_run(ctx, classe)
        if not (run / "results.csv").is_file():
            raise FileNotFoundError(f"results.csv ausente em {run}")
        destino = ctx.saida / "treino" / classe
        destino.mkdir(parents=True, exist_ok=True)
        for arq in [run / "results.csv", run / "args.yaml"] + sorted(run.glob("*.png")):
            if arq.is_file():
                shutil.copy2(arq, destino / arq.name)
        (ctx.saida / "pesos").mkdir(exist_ok=True)
        shutil.copy2(best, ctx.saida / "pesos" / f"{classe}_best.pt")

    # --- Manifesto com o SHA-256 de cada arquivo ---
    manifesto = []
    for arq in sorted(ctx.saida.rglob("*")):
        rel = arq.relative_to(ctx.saida)
        if not arq.is_file() or rel.parts[0] in EXCLUIR_DO_MANIFESTO:
            continue
        if arq.name == "manifesto_saidas.json" or (len(rel.parts) == 1 and arq.name.startswith("log_")):
            continue
        manifesto.append({"arquivo": str(rel), "bytes": arq.stat().st_size, "sha256": sha256_arquivo(arq)})
    (ctx.saida / "manifesto_saidas.json").write_text(
        json.dumps({"config_yolo_final_sha256": hash_config, "arquivos": manifesto}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Manifesto: {len(manifesto)} arquivos em {ctx.saida}")


# =============================================================================
# PRÉVIA CLÁSSICO x YOLO NO VALID (uma classe; não toca no test nem no config)
# =============================================================================

def _valores(metricas: pd.DataFrame, metodo: str, escopo: str, classe: str, dataset: str) -> pd.Series:
    sub = metricas[
        (metricas["metodo"] == metodo)
        & (metricas["escopo"] == escopo)
        & (metricas["classe"] == classe)
        & (metricas["dataset"] == dataset)
    ]
    if sub.empty:
        raise RuntimeError(f"Métricas ausentes: {metodo}/{escopo}/{classe}/{dataset}")
    return sub.set_index("metrica")["valor"]


def tabela_lado_a_lado(metricas: pd.DataFrame, classe: str, metodo_classico: str, metodo_yolo: str) -> pd.DataFrame:
    """Uma linha por dataset da classe e uma final com o geral da classe."""
    datasets = sorted(
        metricas[(metricas["escopo"] == "classe_dataset") & (metricas["classe"] == classe)]["dataset"].unique().tolist()
    )
    alvos = [("classe_dataset", d) for d in datasets] + [("classe", "TODOS")]
    linhas = []
    for escopo, ds in alvos:
        c = _valores(metricas, metodo_classico, escopo, classe, ds)
        y = _valores(metricas, metodo_yolo, escopo, classe, ds)
        reducao = (c["mae"] - y["mae"]) / c["mae"] * 100.0 if c["mae"] > 0 else float("nan")
        linhas.append(
            {
                "dataset": ds if escopo == "classe_dataset" else f"GERAL ({classe})",
                "n_imagens": int(c["n_imagens"]),
                "mae_classico": float(c["mae"]),
                "mae_yolo": float(y["mae"]),
                "reducao_mae_pct": float(reducao),
                "vies_classico": float(c["vies"]),
                "vies_yolo": float(y["vies"]),
                "exato_classico_pct": float(c["acerto_exato_pct"]),
                "exato_yolo_pct": float(y["acerto_exato_pct"]),
                "tol1_classico_pct": float(c["acerto_tol1_pct"]),
                "tol1_yolo_pct": float(y["acerto_tol1_pct"]),
            }
        )
    return pd.DataFrame(linhas)


def cmd_previa(ctx: Contexto) -> None:
    """Subcomando previa: compara clássico e YOLO no valid para uma classe, sem tocar no test nem no config."""
    from ultralytics import YOLO

    classe = ctx.classe
    if classe not in CLASSES:
        raise RuntimeError("`previa` exige --classe <parafuso|porca|arruela>.")
    _, best, _, marcador = caminhos_run(ctx, classe)
    if not marcador.is_file() or not best.is_file():
        raise RuntimeError(f"[{classe}] sem train_concluido.json/best.pt em {best.parent.parent}: treine a classe antes (`treinar --classe {classe}`).")
    if not ARQUIVO_CLASSICO_VALID.is_file():
        raise RuntimeError(f"Previsões clássicas do valid não encontradas: {ARQUIVO_CLASSICO_VALID}")

    saida = ctx.saida / "previa" / classe  # único local de escrita desta função
    saida.mkdir(parents=True, exist_ok=True)
    log(CABECALHO_PREVIA)
    log(f"[{classe}] best.pt: {best}")
    df_gt = verificar_dados()
    gt_cls = linhas_gt_da_classe(df_gt, classe, "valid")
    log(f"[{classe}] {len(gt_cls)} linhas de GT no valid")

    # --- 1. YOLO: inferência (conf 0,05) e varredura de limiares, como no calibrar ---
    det = inferir_deteccoes(YOLO(str(best)), classe, "valid", gt_cls, DATA_FINAL / "yolo", ctx.config)
    salvar_deteccoes(det, saida / f"deteccoes_valid_{classe}.json")
    with tempfile.TemporaryDirectory(prefix="previa_yolo_") as tmp:
        curva, limiar = calibrar_limiar(gt_cls, det, classe, tmp, ctx.metodo)
    curva.to_csv(saida / "curva_limiares.csv", index=False, encoding="utf-8")
    log(f"[{classe}] limiar escolhido (menor MAE no valid) = {limiar:.2f}")

    # --- 2. Avaliação com scripts/avaliar.py (--split valid) dos dois métodos, só nas linhas da classe ---
    prev_yolo = montar_previsoes(gt_cls, det, limiar, ctx.metodo)
    prev_cls = pd.read_csv(ARQUIVO_CLASSICO_VALID, encoding="utf-8")
    prev_cls = prev_cls[(prev_cls["classe"] == classe) & (prev_cls["split"] == "valid")].reset_index(drop=True)
    metodos_cls = prev_cls["metodo"].unique().tolist()
    if len(metodos_cls) != 1:
        raise RuntimeError(f"Esperado exatamente 1 método no CSV clássico, encontrado: {metodos_cls}")
    metodo_cls = str(metodos_cls[0])
    if metodo_cls == ctx.metodo:
        raise RuntimeError("O nome do método clássico coincide com o do YOLO; impossível comparar.")
    gt_cls.to_csv(saida / f"gt_valid_{classe}.csv", index=False, encoding="utf-8")
    prev_cls.to_csv(saida / f"previsoes_valid_{metodo_cls}_{classe}.csv", index=False, encoding="utf-8")
    prev_yolo.to_csv(saida / f"previsoes_valid_{ctx.metodo}_{classe}.csv", index=False, encoding="utf-8")
    juntas = pd.concat([prev_cls[COLUNAS_PREVISOES], prev_yolo[COLUNAS_PREVISOES]], ignore_index=True)
    juntas.to_csv(saida / f"previsoes_valid_comparacao_{classe}.csv", index=False, encoding="utf-8")
    metricas, _, _ = av.avaliar(
        caminho_previsoes=saida / f"previsoes_valid_comparacao_{classe}.csv",
        caminho_gt=saida / f"gt_valid_{classe}.csv",
        nome_execucao="avaliacao_previa",
        split="valid",
        pasta_resultados=saida,
        imprimir=False,
    )

    # --- 3. Tabela lado a lado ---
    tabela = tabela_lado_a_lado(metricas, classe, metodo_cls, ctx.metodo)
    tabela.to_csv(saida / "tabela_previa.csv", index=False, encoding="utf-8")
    fmt = tabela.copy()
    fmt["n_imagens"] = fmt["n_imagens"].astype(int)
    for col in fmt.columns:
        if col not in ("dataset", "n_imagens"):
            fmt[col] = fmt[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
    texto = (
        "=" * 100 + "\n"
        + CABECALHO_PREVIA + "\n"
        + "=" * 100 + "\n"
        + f"classe={classe} | clássico={metodo_cls} | yolo={ctx.metodo} (limiar={limiar:.2f}) | split=valid\n"
        + "redução de MAE = (MAE clássico − MAE YOLO) / MAE clássico; positiva = YOLO melhor. "
        + "Acerto exato e ±1 em % das imagens.\n\n"
        + fmt.to_string(index=False)
        + "\n"
    )
    (saida / "tabela_previa.txt").write_text(texto, encoding="utf-8")
    (saida / "previa_meta.json").write_text(
        json.dumps(
            {
                "cabecalho": CABECALHO_PREVIA,
                "classe": classe,
                "metodo_yolo": ctx.metodo,
                "metodo_classico": metodo_cls,
                "limiar_yolo": limiar,
                "best_pt_sha256": sha256_arquivo(best),
                "previsoes_classico_sha256": sha256_arquivo(ARQUIVO_CLASSICO_VALID),
                "gerado_em_utc": agora_utc(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\n" + texto)
    log(f"Prévia salva em {saida}")


# =============================================================================
# CLI
# =============================================================================

def _parser() -> argparse.ArgumentParser:
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--saida", default=str(SAIDA_PADRAO), help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    comum.add_argument("--modelo", default="yolo26n.pt", help="pesos pré-treinados (padrão: yolo26n.pt)")
    comum.add_argument("--epochs", type=int, default=50)
    comum.add_argument("--patience", type=int, default=10)
    comum.add_argument("--batch", type=int, default=8)
    comum.add_argument("--imgsz", type=int, default=640)
    comum.add_argument("--seed", type=int, default=42)
    comum.add_argument("--max-det", dest="max_det", type=int, default=300)
    comum.add_argument("--fraction", type=float, default=1.0, help="fração do train usada no treino (só para smoke test)")

    p = argparse.ArgumentParser(description="Fase 4A-LOCAL — YOLO por classe (treino, calibração e test).")
    sub = p.add_subparsers(dest="comando", required=True)
    for nome, ajuda in (
        ("treinar", "treina um modelo por classe"),
        ("calibrar", "calibra limiares no valid e congela o config"),
        ("testar", "executa o test uma única vez e salva os entregáveis"),
        ("tudo", "treinar -> calibrar -> testar"),
        ("previa", "prévia clássico x YOLO no VALID para uma classe já treinada (não toca no test nem no config)"),
    ):
        sp = sub.add_parser(nome, parents=[comum], help=ajuda)
        if nome == "previa":
            sp.add_argument("--classe", choices=CLASSES, required=True, help="classe já treinada")
        if nome == "treinar":
            sp.add_argument("--classe", choices=CLASSES, default=None, help="treina apenas esta classe (padrão: as 3, em sequência)")
        if nome in ("treinar", "tudo"):
            sp.add_argument("--retomar", action="store_true", help="retoma treinos interrompidos a partir do last.pt")
        if nome in ("testar", "tudo"):
            sp.add_argument("--hash-esperado", dest="hash_esperado", default=None, help="SHA-256 esperado do config_yolo_final.json (opcional)")
    return p


def main() -> None:
    """Lê os argumentos, abre o log e despacha para o subcomando pedido."""
    args = _parser().parse_args()
    ctx = Contexto(args)
    # A prévia escreve somente em <saida>/previa/<classe>/, inclusive o log.
    pasta_log = ctx.saida / "previa" / args.classe if args.comando == "previa" else ctx.saida
    caminho_log = iniciar_log(pasta_log)
    log(f"Log: {caminho_log}")
    log(f"Comando: {args.comando} | saída: {ctx.saida} | método: {ctx.metodo}")
    try:
        if args.comando == "previa":
            cmd_previa(ctx)
        if args.comando in ("treinar", "tudo"):
            cmd_treinar(ctx)
        if args.comando in ("calibrar", "tudo"):
            cmd_calibrar(ctx)
        if args.comando in ("testar", "tudo"):
            cmd_testar(ctx)
    except BaseException:
        log("ERRO — execução interrompida:\n" + traceback.format_exc())
        raise
    log("Concluído.")


if __name__ == "__main__":
    main()
