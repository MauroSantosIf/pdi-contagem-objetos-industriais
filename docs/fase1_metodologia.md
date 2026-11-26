# Fase 1 — Metodologia de Preparação e Validação dos Datasets

## 1. Introdução

A primeira fase do projeto de Processamento Digital de Imagens (PDI) teve como objetivo organizar, analisar e validar os datasets utilizados para identificação e contagem de objetos industriais.

A etapa foi desenvolvida em Python, utilizando o Google Colab como ambiente de execução e o Google Drive como armazenamento dos arquivos.

O trabalho concentrou-se na preparação das informações necessárias para as fases seguintes, especialmente o pré-processamento, o reconhecimento de objetos e a avaliação dos resultados.

**Responsável pela Fase 1:** Vivian Beatriz.

## 2. Objetivos

Os principais objetivos desta etapa foram:

- Organizar quatro datasets de objetos industriais.
- Identificar os formatos e as classes das anotações.
- Verificar a integridade dos arquivos.
- Detectar inconsistências nos rótulos.
- Investigar imagens duplicadas.
- Produzir versões corrigidas ou filtradas, quando necessário.
- Gerar o Ground Truth para cada imagem.
- Documentar os resultados e as limitações identificadas.

O escopo desta fase não incluiu treinamento de modelos de inteligência artificial nem aplicação de técnicas de pré-processamento.

## 3. Ambiente e ferramentas

| Tecnologia | Utilização |
|---|---|
| Python | Implementação das rotinas de análise |
| Google Colab | Execução do notebook |
| Google Drive | Armazenamento dos datasets e resultados |
| Pandas | Organização e exportação dos dados |
| NumPy | Suporte a operações numéricas |
| PyYAML | Leitura das configurações dos datasets |
| hashlib | Cálculo dos hashes SHA-256 |
| zipfile | Leitura e extração dos arquivos ZIP |
| JSON e CSV | Registro dos resultados |

O notebook principal é `fase1_multidataset_consolidada.ipynb`, localizado na pasta `notebooks/` do repositório.

## 4. Fluxo metodológico

A metodologia foi organizada em seis etapas:

1. Inventário e seleção dos datasets.
2. Leitura das imagens e anotações.
3. Validação estrutural dos rótulos.
4. Auditoria de duplicatas e tratamento de inconsistências.
5. Geração do Ground Truth.
6. Consolidação e verificação dos resultados.

Esse processo foi executado individualmente para cada dataset, respeitando suas classes e características.

## 5. Organização e seleção dos datasets

Foram selecionados quatro conjuntos de dados:

| Dataset | Versão final | Imagens |
|---|---|---:|
| Roboflow Screws | `revisado.zip` | 463 |
| NutsNBolts | `filtrado.zip` | 107 |
| Washers Hardee | `deduplicado.zip` | 402 |
| Rebar Counting V1 | `original.zip` | 3 |

Os arquivos originais foram mantidos para preservar os dados de origem e permitir a rastreabilidade das modificações.

O notebook utiliza explicitamente as versões selecionadas, evitando retornar automaticamente ao arquivo original quando uma versão tratada já foi aprovada.

## 6. Leitura das imagens e anotações

Os datasets utilizam o padrão de organização YOLO, com imagens e arquivos de anotação associados.

A estrutura convencional contém:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

O arquivo `data.yaml` define as classes utilizadas pelo dataset.

Para cada imagem identificada, o programa procura seu arquivo correspondente na pasta `labels/`.

Os identificadores de classe são interpretados separadamente em cada dataset. Assim, a classe `0` de um conjunto não é considerada automaticamente equivalente à classe `0` de outro.

### 6.1. Formatos de anotação

Foram considerados dois formatos:

**YOLO Detection**

Cada linha representa um objeto utilizando:

```text
classe centro_x centro_y largura altura
```

**YOLO Segmentation**

Cada linha representa um objeto por meio de um conjunto de coordenadas que descrevem um polígono:

```text
classe x1 y1 x2 y2 x3 y3 ...
```

As coordenadas são normalizadas em relação às dimensões da imagem.

## 7. Validação estrutural das anotações

A validação foi realizada automaticamente por funções implementadas em Python.

### 7.1. Verificação de classes

Cada anotação foi comparada com as classes declaradas no respectivo `data.yaml`.

Identificadores inexistentes foram considerados inválidos.

### 7.2. Verificação de coordenadas

As anotações foram analisadas para identificar:

- Valores não numéricos.
- Coordenadas não finitas.
- Coordenadas fora dos limites permitidos.
- Caixas delimitadoras com largura ou altura inválida.
- Polígonos com quantidade inadequada de coordenadas.
- Polígonos sem extensão válida nos eixos.

Para as caixas de detecção, também foi verificado se seus limites permaneciam dentro da imagem.

### 7.3. Correspondência entre imagens e rótulos

O programa verificou:

- Imagens sem arquivos de anotação.
- Arquivos de anotação sem imagem correspondente.
- Distribuição de imagens entre `train`, `valid` e `test`.
- Quantidade de anotações válidas por imagem.

As inconsistências identificadas foram registradas para auditoria.

## 8. Auditoria de duplicatas

Foi utilizado o algoritmo SHA-256 para calcular um identificador a partir do conteúdo binário de cada imagem.

Imagens com hashes iguais foram consideradas duplicatas exatas.

O método permite localizar arquivos binariamente idênticos, mesmo quando possuem nomes diferentes.

Entretanto, ele não identifica necessariamente imagens visualmente semelhantes que tenham sofrido transformações, recortes ou alterações de compressão.

Também foi realizada uma análise de nomes de origem para identificar possíveis relações entre imagens pertencentes a diferentes conjuntos de treinamento, validação e teste.

Esses nomes foram tratados como indícios de possível relação entre imagens, e não como prova definitiva de duplicidade.

## 9. Tratamento das inconsistências

As decisões de tratamento foram específicas para cada dataset.

### 9.1. Parafusos

Foram identificadas inconsistências geométricas nas anotações.

O tratamento incluiu:

- Ajuste de duas caixas delimitadoras aos limites das imagens.
- Exclusão de três anotações inválidas.
- Preservação das demais anotações.

A versão resultante foi armazenada em `revisado.zip`.

Resultado: 463 imagens e 3.230 objetos anotados.

### 9.2. NutsNBolts

Foram identificadas 32 caixas delimitadoras problemáticas, distribuídas em 23 imagens.

A equipe optou por excluir essas 23 imagens de uma versão derivada, em vez de modificar individualmente suas anotações.

O conjunto resultante foi armazenado em `filtrado.zip`.

Resultado: 107 imagens e 2.832 objetos anotados.

Essa decisão reduziu o tamanho do dataset e deve ser considerada na avaliação futura dos modelos.

### 9.3. Arruelas

Foi realizada uma auditoria de imagens duplicadas.

Inicialmente, três cópias redundantes foram removidas, produzindo `revisado.zip`.

Posteriormente, identificaram-se 245 grupos adicionais de imagens binariamente idênticas, contendo 331 cópias excedentes.

As anotações associadas às imagens de cada grupo também foram comparadas, confirmando que eram idênticas.

Foi preservada uma imagem e seu respectivo rótulo por grupo.

O conjunto final foi armazenado em `deduplicado.zip`.

Resultado: 402 imagens e 3.949 objetos anotados, sem duplicatas exatas restantes.

### 9.4. Vergalhões

O dataset apresentou três imagens e 26 objetos anotados, sem inconsistências estruturais identificadas.

Entretanto, todas as imagens pertencem ao conjunto `train`.

Como não existem imagens nos conjuntos `valid` e `test`, esse dataset foi classificado como exploratório.

Nenhuma divisão artificial das três imagens foi realizada.

## 10. Construção do Ground Truth

O Ground Truth foi gerado a partir das anotações consideradas válidas em cada dataset.

Cada linha válida de anotação corresponde a uma instância contabilizada.

Para cada imagem, foram registrados:

- Identificador do dataset.
- Caminho da imagem.
- Conjunto de origem.
- Presença do arquivo de anotação.
- Quantidade de anotações válidas.
- Quantidade de anotações inválidas.
- Quantidade de objetos por classe.
- Necessidade de revisão.

Os resultados foram organizados em tabelas utilizando a biblioteca Pandas.

A contagem de objetos foi obtida por meio da soma das anotações válidas.

**Limitação metodológica:** como as contagens foram derivadas dos rótulos existentes, o processo não garante que todos os objetos visualmente presentes estejam anotados. Uma auditoria visual complementar é necessária para avaliar a completude das anotações.

## 11. Geração dos relatórios

O notebook consolidado produz arquivos destinados à rastreabilidade e à análise posterior.

Entre os principais resultados estão:

| Arquivo | Finalidade |
|---|---|
| `ground_truth.csv` | Quantidades anotadas por imagem |
| `problemas_anotacoes.csv` | Inconsistências estruturais encontradas |
| `duplicatas_exatas.csv` | Grupos de imagens binariamente idênticas |
| `suspeitas_entre_splits.csv` | Indícios de imagens relacionadas entre conjuntos |
| `resumo.json` | Estatísticas e informações do dataset |
| `consolidado.csv` | Resumo geral dos quatro datasets |
| `manifesto_execucao.json` | Registro das versões e arquivos processados |
| `criterios_aceitacao.csv` | Resultado das verificações finais |

O manifesto também registra o SHA-256 dos arquivos ZIP utilizados, auxiliando na identificação das versões efetivamente processadas.

Cada execução possui um diretório próprio no Google Drive para evitar a substituição dos relatórios de execuções anteriores.

## 12. Resultados da validação

A execução consolidada realizada em 26/11/2026 apresentou os seguintes resultados:

| Dataset | Imagens | Objetos | Problemas estruturais | Duplicatas exatas |
|---|---:|---:|---:|---:|
| Parafusos | 463 | 3.230 | 0 | 0 |
| NutsNBolts | 107 | 2.832 | 0 | 0 |
| Arruelas | 402 | 3.949 | 0 | 0 |
| Vergalhões | 3 | 26 | 0 | 0 |
| **Total** | **975** | **10.037** | **0** | **0** |

Todos os datasets atenderam aos critérios de inventário, integridade estrutural e ausência de duplicatas binárias.

No entanto, o conjunto de vergalhões permaneceu classificado como exploratório por não possuir imagens suficientes para treinamento e avaliação adequados.

## 13. Limitações identificadas

Mesmo após a aprovação estrutural, permanecem algumas limitações:

1. O Ground Truth foi derivado das anotações existentes, sem conferência visual completa.
2. O SHA-256 não identifica necessariamente imagens semelhantes com pequenas alterações.
3. O dataset de arruelas apresenta dois grupos com nomes de origem compartilhados entre diferentes conjuntos, exigindo cautela na avaliação de possível vazamento de dados.
4. O dataset NutsNBolts possui apenas cinco imagens para validação e cinco para teste.
5. O dataset de vergalhões apresenta apenas três imagens, exclusivamente no conjunto de treinamento.
6. Os datasets utilizam classes e formatos de anotação que ainda precisam ser padronizados para uma eventual integração.

Essas limitações deverão ser consideradas durante as próximas fases.

## 14. Reprodutibilidade

O procedimento pode ser reproduzido por meio do notebook:

`notebooks/fase1_multidataset_consolidada.ipynb`

Para sua execução, é necessário:

1. Disponibilizar os datasets e suas versões selecionadas no Google Drive.
2. Abrir o notebook no Google Colab.
3. Autorizar o acesso ao Google Drive.
4. Executar as células na ordem apresentada.
5. Conferir os relatórios e os critérios de aceitação.

O notebook não modifica os arquivos ZIP utilizados como entrada durante a execução consolidada.

## 15. Conclusão

A Fase 1 permitiu organizar e validar estruturalmente os quatro datasets selecionados para o projeto de contagem de objetos industriais.

Foram processadas 975 imagens, totalizando 10.037 instâncias anotadas.

As versões finais não apresentaram inconsistências estruturais nem duplicatas binariamente idênticas.

Além disso, foram documentadas as decisões de correção, filtragem e deduplicação, preservando os arquivos originais e possibilitando a reprodução das análises.

Com essa etapa concluída, os dados e relatórios ficam disponíveis para a **Fase 2 — Pré-processamento de Imagens**, respeitando as limitações identificadas e preservando a correspondência entre imagens e anotações.

---

*Projeto acadêmico de Processamento Digital de Imagens — Fase 1: Organização, Validação e Ground Truth.*
