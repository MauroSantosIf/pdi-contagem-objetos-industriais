# Contagem de objetos em imagens industriais

Projeto da disciplina de Processamento Digital de Imagens (PDI). O trabalho compara duas formas de contar parafusos, porcas e arruelas em uma imagem: uma pipeline clássica de processamento de imagens, feita com OpenCV, e detectores YOLO, com um detector por classe.

Repositório: https://github.com/MauroSantosIf/pdi-contagem-objetos-industriais

## 1. Objetivo

Contar, para cada imagem, quantos objetos de cada classe (`parafuso`, `porca`, `arruela`) ela contém, e medir o erro de cada método contra o GT (contagem de referência, derivada das anotações dos datasets).

As duas abordagens passam pelo mesmo avaliador (`scripts/avaliar.py`), com as mesmas divisões `train`, `valid` e `test`. As métricas são MAE (erro absoluto médio, em objetos por imagem), viés (previsto menos real; positivo quer dizer que o método superestima), acerto exato e acerto ±1 (percentual de imagens com erro 0 e com erro de no máximo 1 objeto).

O dataset de vergalhões (`datasets/vergalhoes/`) faz parte do inventário inicial, mas tem só 3 imagens e não entra na base final nem nas comparações.

## 2. Resumo dos resultados

Resultados no `test`: 96 imagens e 117 avaliações (imagem por classe). A pipeline clássica foi calibrada por dataset e o detector YOLO por classe, ambos só com `train` e `valid`. O `test` foi usado uma única vez em cada método, com a configuração já congelada.

Fonte dos números desta seção: `resultados/yolo/comparacao/comparacao_metricas_sem_exclusoes.csv`. Os valores da pipeline clássica conferem com `resultados/avaliacao/classico_test/metricas.csv`, e os do YOLO com `resultados/yolo/avaliacao/yolo26s_test/metricas.csv`.

### Por classe e no geral

| Escopo | n | MAE clássica | MAE YOLO | Viés clássica | Viés YOLO | Acerto exato clássica (%) | Acerto exato YOLO (%) | Acerto ±1 clássica (%) | Acerto ±1 YOLO (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Geral | 117 | 4,03 | 0,85 | 0,43 | -0,22 | 23,93 | 62,39 | 45,30 | 82,91 |
| Arruela | 62 | 6,11 | 1,13 | 0,85 | -0,52 | 22,58 | 61,29 | 35,48 | 74,19 |
| Parafuso | 40 | 1,55 | 0,45 | -0,30 | -0,05 | 27,50 | 62,50 | 62,50 | 95,00 |
| Porca | 15 | 2,07 | 0,80 | 0,60 | 0,53 | 20,00 | 66,67 | 40,00 | 86,67 |

### Por classe e dataset (MAE)

| Classe | Dataset | n | MAE clássica | MAE YOLO |
|---|---|---:|---:|---:|
| Arruela | `nutsnbolts` | 6 | 1,67 | 3,17 |
| Arruela | `washers_hardee` | 56 | 6,59 | 0,91 |
| Parafuso | `nutsnbolts` | 6 | 2,00 | 0,67 |
| Parafuso | `roboflow_screws` | 25 | 1,28 | 0,32 |
| Parafuso | `screws_and_nuts` | 9 | 2,00 | 0,67 |
| Porca | `nutsnbolts` | 6 | 2,50 | 1,67 |
| Porca | `screws_and_nuts` | 9 | 1,78 | 0,22 |

O YOLO tem MAE menor em todos os recortes, menos nas arruelas de `nutsnbolts`. Esse caso é discutido na seção 7.

### Teste de Wilcoxon pareado (erro absoluto por imagem e classe)

Fonte: `resultados/yolo/comparacao/pareada_wilcoxon_sem_exclusoes.csv`.

| Classe | n | YOLO erra menos | Empate | Clássica erra menos | p-valor |
|---|---:|---:|---:|---:|---:|
| Todas | 117 | 76 | 29 | 12 | menor que 0,0001 |
| Arruela | 62 | 43 | 13 | 6 | menor que 0,0001 |
| Parafuso | 40 | 24 | 12 | 4 | 0,0002 |
| Porca | 15 | 9 | 4 | 2 | 0,0796 |

O arquivo grava o p-valor arredondado a 4 casas, por isso aparece 0,0 para "Todas" e "Arruela". Para porca, o p-valor de 0,0796 não permite afirmar diferença significativa a 5%.

### Erro na contagem total por imagem

Soma das três classes em cada imagem. Fonte: `resultados/yolo/comparacao/total_por_imagem_sem_exclusoes.csv`.

| Dataset | Imagens | Peças por imagem (média) | Erro médio clássica | Erro médio YOLO |
|---|---:|---:|---:|---:|
| `nutsnbolts` | 6 | 27,83 | 3,50 | 2,50 |
| `roboflow_screws` | 25 | 6,72 | 1,28 | 0,32 |
| `screws_and_nuts` | 9 | 10,67 | 2,67 | 0,89 |
| `washers_hardee` | 56 | 8,98 | 6,59 | 0,91 |

### Configurações congeladas

- Pipeline clássica: `resultados/classico/config_final.json`, com SHA-256 `049ff32f47663757df09fbaca16da09a49b74d535a45df668eac8ed721cbe2fd`. Calibração por dataset, com busca em grade de 32 a 48 combinações.
- Detector YOLO: `resultados/yolo/config_yolo_final.json`, com SHA-256 `2e4858358bb3c04ceeb409e75ac3c0e819d88cf28d62c134dfab7fec22df590d`. Modelo `yolo26s`, no máximo 100 épocas com `patience = 20`, `batch = 16`, `imgsz = 640`, semente 42. Limiares de confiança calibrados no `valid`: parafuso 0,30, porca 0,15 e arruela 0,80.

## 3. Organização do repositório

```text
pdi-contagem-objetos-industriais/
├── README.md
├── datasets/                 # ZIPs originais e tratados dos datasets (formato YOLO)
│   ├── arruelas/washers_hardee/
│   ├── parafusos/roboflow_screws/
│   ├── porcas/nutsnbolts/
│   ├── porcas/screws_and_nuts/
│   └── vergalhoes/rebar_counting_v1/
├── scripts/
│   ├── preparar_dados.py             # Fase 1: gera data_final/ e o GT de contagem
│   ├── avaliar.py                    # Fase 2: avaliador comum aos dois métodos
│   ├── testar_avaliar.py             # Fase 2: testes do avaliador
│   ├── preparar_auditoria_teste.py   # Fase 2: material para auditar o GT do test
│   ├── classico.py                   # Fase 3: pipeline clássica
│   └── yolo_local.py                 # Fase 4: treino, calibração e test do YOLO em máquina local
├── notebooks/
│   ├── etapa3_pipeline_classica.ipynb      # apresenta os resultados da pipeline clássica
│   ├── etapa4_yolo_colab.ipynb             # treino e avaliação do YOLO no Colab (resultado oficial)
│   ├── etapa4b_comparacao_colab.ipynb      # comparação clássica x YOLO e demonstração
│   ├── fase1_multidataset_consolidada.ipynb
│   └── fase2_preprocessamento_multidataset_sincronizado.ipynb
├── docs/                     # metodologia e inventário dos datasets
├── resultados/
│   ├── avaliacao/            # métricas da pipeline clássica (valid, test e test com exclusões)
│   ├── auditoria/            # planilha de auditoria do GT do test e lista de exclusões
│   ├── classico/             # configuração congelada, previsões, histogramas e figuras
│   ├── yolo/                 # resultado oficial do YOLO (Colab): config, previsões, métricas, comparação
│   └── yolo_local_parcial/   # execução local anterior e incompleta (ignorada pelo Git)
├── data_final/               # gerado por preparar_dados.py (ignorado pelo Git)
└── data_final.zip            # idem
```

`data_final/`, `data_final.zip`, `data_extracted/`, `resultados/yolo/pesos/`, `resultados/yolo/runs/` e `resultados/yolo_local_parcial/` estão no `.gitignore`. Os pesos do YOLO ficam em `resultados/yolo/pesos/` e não vão para o Git.

## 4. Datasets

A base final (`data_final/`) usa quatro datasets. As classes de cada um são mapeadas para `parafuso`, `porca` e `arruela`. Contagens de `data_final/contagens_gt.csv`, depois do agrupamento de imagens parecidas e do sorteio por grupo:

| Dataset | Classes usadas | `train` | `valid` | `test` | Imagens | Objetos |
|---|---|---:|---:|---:|---:|---:|
| `roboflow_screws` | parafuso | 335 | 25 | 25 | 385 | 2.715 |
| `nutsnbolts` | parafuso, porca, arruela | 78 | 6 | 6 | 90 | 2.370 |
| `screws_and_nuts` | parafuso, porca | 97 | 9 | 9 | 115 | 1.984 |
| `washers_hardee` | arruela | 283 | 56 | 56 | 395 | 3.911 |
| **Total** | | **793** | **96** | **96** | **985** | **10.980** |

O GT tem 1.280 linhas (dataset, divisão, imagem e classe). Cada detector YOLO treina com as imagens do `train` dos datasets que rotulam a classe dele: parafuso 510 imagens, porca 175 e arruela 361.

As divisões foram sorteadas por grupos de imagens parecidas (nome de origem e dHash), para não haver vazamento entre `train`, `valid` e `test`. O detalhe está em `scripts/preparar_dados.py` e em `data_final/manifesto.json`.

Documentação de cada dataset:

- [Parafusos (Roboflow Screws)](datasets/parafusos/README.md)
- [Porcas (NutsNBolts e Screws and Nuts)](datasets/porcas/README.md)
- [Arruelas (Washers Hardee)](datasets/arruelas/README.md)
- [Vergalhões (Rebar Counting V1)](datasets/vergalhoes/README.md)
- [Inventário geral](docs/datasets.md)

A utilização e a redistribuição dos arquivos devem respeitar as licenças das fontes.

## 5. Fases do projeto e como reproduzir

Os comandos abaixo partem da raiz do repositório, com o ambiente virtual em `.venv`. O repositório não tem `requirements.txt`. As bibliotecas usadas estão nos `import` dos scripts: OpenCV, NumPy, Pandas, PyYAML, Matplotlib e SciPy, além de `torch` e `ultralytics` para o YOLO.

### Fase 1: preparação dos dados

`scripts/preparar_dados.py` lê os ZIPs de `datasets/`, unifica as classes, agrupa imagens parecidas, sorteia as divisões com semente fixa, grava `data_final/` e o GT (`data_final/contagens_gt.csv`) e compacta tudo em `data_final.zip`. Apaga e recria essas saídas a cada execução, com o mesmo resultado.

```bash
.venv/bin/python scripts/preparar_dados.py
```

Os ZIPs de entrada (`revisado.zip`, `filtrado.zip`, `deduplicado.zip`, `original.zip`) vêm da organização e validação dos datasets feita antes, descrita em `docs/fase1_metodologia.md`.

### Fase 2: avaliador

`scripts/avaliar.py` recebe um CSV de previsões (`metodo`, `dataset`, `split`, `imagem`, `classe`, `contagem`), casa cada linha com o GT de forma estrita e calcula as métricas. Os testes ficam em `scripts/testar_avaliar.py`. A auditoria do GT do `test` usa `scripts/preparar_auditoria_teste.py`, que desenha as caixas sobre as imagens e cria a planilha em `resultados/auditoria/`.

```bash
.venv/bin/python scripts/testar_avaliar.py
.venv/bin/python scripts/avaliar.py --previsoes <previsoes.csv> --nome-execucao <nome> --split valid
.venv/bin/python scripts/preparar_auditoria_teste.py
```

### Fase 3: pipeline clássica

`scripts/classico.py` implementa 7 etapas: pré-processamento, segmentação, morfologia, separação por watershed, componentes conectados, classificação por regras e contagem por classe. A calibração é por dataset e usa só `train` e `valid`. O notebook `notebooks/etapa3_pipeline_classica.ipynb` mostra os resultados.

```bash
.venv/bin/python scripts/classico.py --modo calibrar        # calibra e congela config_final.json
.venv/bin/python scripts/classico.py --modo avaliar-valid   # reaplica a configuração no valid
.venv/bin/python scripts/classico.py --modo avaliar-test --hash-esperado <SHA256>   # test, uma única vez
```

O modo `calibrar` sobrescreve `resultados/classico/`. O resultado oficial já está gravado ali e não deve ser regerado para ajustar nada.

### Fase 4: detector YOLO

Um detector `yolo26s` por classe. A contagem de uma imagem é o número de caixas com confiança maior ou igual ao limiar da classe. O limiar é escolhido no `valid`, pelo menor MAE (empate: menor |viés|, depois menor limiar), e congelado antes do `test`.

O resultado oficial vem do Colab, com GPU T4:

1. Suba `data_final.zip` e uma cópia idêntica de `scripts/avaliar.py` para `MyDrive/PDI_Contagem_Objetos/etapa4/`.
2. Execute `notebooks/etapa4_yolo_colab.ipynb` de cima para baixo. As saídas ficam em `etapa4/saidas/`.
3. Copie `etapa4/saidas/` para `resultados/yolo/`. O arquivo `manifesto_saidas.json` lista o SHA-256 de cada arquivo.

`scripts/yolo_local.py` repete o mesmo fluxo em máquina local, com padrões de treino mais leves (`yolo26n`, 50 épocas, `patience = 10`, `batch = 8`). Por isso, o resultado dele não substitui o do Colab.

```bash
.venv/bin/python scripts/yolo_local.py tudo      # treinar, calibrar e testar
.venv/bin/python scripts/yolo_local.py previa --classe porca   # compara clássica e YOLO no valid, sem tocar no test
```

### Fase 4B: comparação

`notebooks/etapa4b_comparacao_colab.ipynb` junta as previsões do `test` dos dois métodos, calcula as métricas, faz a análise pareada com Wilcoxon, mostra as 10 piores imagens do YOLO e gera os gráficos. Precisa de `resultados/classico/previsoes_test.csv` em `etapa4/classico/` no Drive, além do que a Fase 4 já deixou lá. Só lê resultados congelados. As saídas estão em `resultados/yolo/comparacao/`.

### Estudo anterior de pré-processamento

Os notebooks `fase1_multidataset_consolidada.ipynb` e `fase2_preprocessamento_multidataset_sincronizado.ipynb` e os documentos `docs/fase1_*.md` e `docs/fase2_*.md` registram a organização e a validação dos datasets e um estudo exploratório de pré-processamento (escala de cinza, mediana, correção de iluminação, equalização e CLAHE). A numeração dos documentos antigos é diferente da dos scripts: lá, a Fase 2 é o pré-processamento, e nos scripts a Fase 2 é o avaliador. A base final não usa as imagens pré-processadas por esse estudo, e a pipeline clássica tem o pré-processamento próprio dela (etapa 1).

Esses notebooks rodam no Google Colab, com os ZIPs de `datasets/` no Drive, em `MyDrive/PDI_Contagem_Objetos/`. Para rodar a partir de um clone do repositório, é preciso adaptar os caminhos.

## 6. Documentação

- [Inventário dos datasets](docs/datasets.md)
- [Metodologia da organização e validação dos datasets](docs/fase1_metodologia.md)
- [Metodologia do estudo de pré-processamento](docs/fase2_metodologia.md)
- [Resultados do estudo de pré-processamento](docs/fase2_resultados.md)
- [Notebooks](notebooks/README.md)

## 7. Limitações e observações

1. O detector YOLO usa um limiar por classe, calibrado no `valid`. Na arruela, o `valid` é dominado por `washers_hardee` (56 das 62 imagens). O limiar de 0,80 escolhido assim piorou as arruelas de `nutsnbolts`, que são o único recorte em que o YOLO ficou pior que a pipeline clássica (MAE 3,17 contra 1,67, em 6 imagens).
2. Nessas imagens, várias detecções têm confiança praticamente igual ao limiar. Nas detecções congeladas do `test`, duas imagens de `nutsnbolts` têm uma detecção de arruela com confiança 0,791, logo abaixo de 0,80 (`resultados/yolo/deteccoes/deteccoes_test_arruela.json`). Na releitura feita no notebook 4B, imagem por imagem, apareceram detecções em 0,800 e 0,819. Em 2 imagens do `test`, a contagem refeita dessa forma diferiu da congelada em 1 a 2 arruelas. Essas confianças da releitura vieram da execução no Colab, cuja saída não foi salva no notebook. O resultado oficial é a execução congelada, em lote.
3. As duas abordagens não foram calibradas no mesmo nível. A pipeline clássica foi calibrada por dataset e o YOLO por classe.
4. Para porca, a redução do erro não foi estatisticamente significativa no teste de Wilcoxon (p = 0,0796), porque o `test` tem só 15 avaliações dessa classe.
5. O GT do `test` não passou por auditoria manual. A planilha `resultados/auditoria/auditoria_teste.csv` está com `gt_completo` em branco e `exclusoes_teste.csv` só tem o cabeçalho. Algumas imagens de `washers_hardee` parecem ter arruelas sem caixa.
6. `washers_hardee` contém imagens de bancos de fotos e um infográfico, fora do domínio industrial.

Outras observações:

- `nutsnbolts` tem só 6 imagens em `valid` e 6 em `test`, o que deixa as métricas desse dataset instáveis.
- Os notebooks do repositório estão sem as saídas da execução no Colab. A evidência da execução são os arquivos de `resultados/`.
- A validação estrutural dos datasets não substitui uma inspeção visual completa das anotações.
- A identificação de duplicatas por SHA-256 não detecta imagens apenas parecidas. Por isso, a Fase 1 também agrupa por nome de origem e dHash.

---

**Projeto:** Contagem de Objetos em Imagens Industriais  
**Disciplina:** Processamento Digital de Imagens
