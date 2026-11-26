# Notebooks do projeto

Cada notebook cobre uma parte do projeto. Os scripts que fazem o trabalho pesado ficam em `scripts/`.

| Notebook | O que faz | Onde roda |
|---|---|---|
| `fase1_multidataset_consolidada.ipynb` | Organiza e valida os datasets e gera o GT da organização inicial. | Google Colab |
| `fase2_preprocessamento_multidataset_sincronizado.ipynb` | Estudo exploratório de pré-processamento (cinza, mediana, iluminação, equalização, CLAHE). | Google Colab |
| `etapa3_pipeline_classica.ipynb` | Mostra os resultados, as figuras e as métricas da pipeline clássica (`scripts/classico.py`). | Local, com o `.venv` do projeto |
| `etapa4_yolo_colab.ipynb` | Treina um detector YOLO por classe, calibra o limiar no `valid` e roda o `test` uma única vez. | Google Colab, com GPU |
| `etapa4b_comparacao_colab.ipynb` | Compara a pipeline clássica com o detector YOLO no `test` e inclui uma demonstração com uma imagem qualquer. | Google Colab |

Cada célula de código dos notebooks `etapa3`, `etapa4` e `etapa4b` é precedida por uma célula de texto com o título "Célula N" e uma explicação curta.

Os notebooks `etapa3`, `etapa4` e `etapa4b` estão sem saídas no repositório. Os resultados das execuções estão em `resultados/`.

Os notebooks `fase1_*` e `fase2_*` pertencem à organização inicial do projeto e seguem a numeração de fases dos documentos em `docs/`. Nos scripts, a Fase 2 é o avaliador, e não o pré-processamento. O README da raiz explica a diferença.
