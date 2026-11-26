# Datasets — Contagem de Objetos Industriais

## 1. Objetivo

Este documento apresenta os datasets utilizados no projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais**.

O projeto tem como objetivo desenvolver uma solução capaz de identificar e contabilizar diferentes objetos industriais em imagens, incluindo:

- Parafusos;
- Porcas;
- Arruelas;
- Vergalhões.

Na **Fase 1 — Organização, Validação e Ground Truth**, foram analisados quatro datasets, com o objetivo de identificar problemas nas anotações, verificar a integridade dos arquivos, remover dados redundantes quando necessário e gerar referências para as próximas etapas.

Os arquivos originais foram preservados. As correções e filtragens foram realizadas em versões derivadas dos datasets.

---

## 2. Resumo dos datasets

| Dataset | Categoria | Imagens | Objetos anotados | Versão utilizada |
|---|---|---:|---:|---|
| Roboflow Screws | Parafusos | 463 | 3.230 | `revisado.zip` |
| NutsNBolts | Porcas, parafusos e arruelas | 107 | 2.832 | `filtrado.zip` |
| Washers Hardee | Arruelas | 402 | 3.949 | `deduplicado.zip` |
| Rebar Counting V1 | Vergalhões | 3 | 26 | `original.zip` |
| **Total** | | **975** | **10.037** | |

Os quantitativos representam as instâncias anotadas nos arquivos de cada dataset.

É importante observar que os datasets apresentam características distintas, incluindo diferentes classes, formatos de anotação e distribuições entre treinamento, validação e teste.

Os dados foram validados estruturalmente, mas ainda não passaram por uma revisão visual exaustiva de todas as anotações.

---

## 3. Dataset de parafusos — Roboflow Screws

### 3.1. Identificação

- **Categoria:** Parafusos
- **Identificador:** `parafusos/roboflow_screws`
- **Fonte:** Roboflow
- **Versão utilizada:** `revisado.zip`
- **Total de imagens:** 463
- **Total de objetos anotados:** 3.230
- **Formato:** YOLO Detection e YOLO Segmentation

### 3.2. Classes

O dataset possui quatro classes:

| ID | Classe | Objetos anotados |
|---|---|---:|
| 0 | Parafuso D2 4.0x25 C4 | 844 |
| 1 | Parafuso D3 35x30 c6 | 805 |
| 2 | Parafuso D4 4.5x16 C16 | 760 |
| 3 | parafuso D1 35x12 c2 | 821 |
| **Total** | | **3.230** |

### 3.3. Distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 387 |
| Validação (`valid`) | 36 |
| Teste (`test`) | 40 |
| **Total** | **463** |

### 3.4. Tratamento realizado

Durante a auditoria das anotações, foram identificadas inconsistências em determinadas caixas delimitadoras.

As correções realizadas foram:

1. Ajuste de duas caixas delimitadoras aos limites válidos da imagem.
2. Exclusão de três anotações inválidas.
3. Preservação das imagens e das demais anotações válidas.
4. Geração da versão derivada `revisado.zip`.

O arquivo `original.zip` foi preservado.

### 3.5. Resultado da validação

- Imagens processadas: **463**
- Anotações válidas: **3.230**
- Problemas estruturais restantes: **0**
- Grupos de duplicatas exatas: **0**
- Situação: **validado estruturalmente**

---

## 4. Dataset NutsNBolts — Porcas, parafusos e arruelas

### 4.1. Identificação

- **Categoria de armazenamento:** Porcas
- **Identificador:** `porcas/nutsnbolts`
- **Fonte:** Roboflow Universe
- **Link:** https://universe.roboflow.com/tktk-hodc3/nutsnbolts
- **Versão utilizada:** `filtrado.zip`
- **Total de imagens:** 107
- **Total de objetos anotados:** 2.832
- **Formato:** YOLO Detection

Embora esteja armazenado na categoria `porcas`, esse dataset contém três classes distintas.

### 4.2. Classes

| ID | Classe | Objetos anotados |
|---|---|---:|
| 0 | `bolt` — Parafuso | 1.564 |
| 1 | `nut` — Porca | 725 |
| 2 | `washer` — Arruela | 543 |
| **Total** | | **2.832** |

### 4.3. Distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 97 |
| Validação (`valid`) | 5 |
| Teste (`test`) | 5 |
| **Total** | **107** |

### 4.4. Tratamento realizado

A versão original possuía:

- 130 imagens;
- 3.408 anotações válidas;
- 32 caixas delimitadoras fora dos limites permitidos;
- 23 imagens afetadas por essas inconsistências.

Como decisão técnica, foram removidas da versão derivada as 23 imagens com problemas de anotação.

Essa decisão evitou a utilização de referências geometricamente inválidas nas próximas etapas.

O arquivo `original.zip` foi preservado, e o dataset filtrado foi armazenado em `filtrado.zip`.

### 4.5. Resultado da validação

- Imagens processadas: **107**
- Anotações válidas: **2.832**
- Problemas estruturais restantes: **0**
- Grupos de duplicatas exatas: **0**
- Situação: **validado estruturalmente**

**Limitação:** os conjuntos de validação e teste possuem apenas cinco imagens cada. Portanto, as métricas de desempenho de um futuro modelo poderão apresentar elevada variabilidade.

---

## 5. Dataset de arruelas — Washers Hardee

### 5.1. Identificação

- **Categoria:** Arruelas
- **Identificador:** `arruelas/washers_hardee`
- **Fonte:** Roboflow
- **Versão utilizada:** `deduplicado.zip`
- **Total de imagens:** 402
- **Total de objetos anotados:** 3.949
- **Formato:** YOLO Detection

### 5.2. Classes

| ID | Classe |
|---|---|
| 0 | `washer` — Arruela |

### 5.3. Distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 280 |
| Validação (`valid`) | 80 |
| Teste (`test`) | 42 |
| **Total** | **402** |

### 5.4. Tratamento realizado

O dataset original apresentava:

- 736 imagens;
- 7.507 objetos anotados;
- Nenhum problema estrutural nas anotações.

Durante as auditorias, foram identificadas imagens exatamente duplicadas.

O processo de limpeza ocorreu em duas etapas.

**Primeira etapa — Revisão inicial**

Foram removidas três cópias redundantes, produzindo a versão `revisado.zip`, com 733 imagens.

**Segunda etapa — Deduplicação completa**

Uma nova auditoria identificou:

- 245 grupos de imagens binariamente idênticas;
- 331 cópias excedentes;
- Anotações idênticas em todos os grupos;
- Nenhuma duplicata exata entre conjuntos distintos.

Foi mantida uma imagem e sua anotação por grupo, removendo apenas as cópias redundantes.

O resultado foi armazenado em `deduplicado.zip`.

### 5.5. Resultado da validação

- Imagens processadas: **402**
- Anotações válidas: **3.949**
- Problemas estruturais restantes: **0**
- Grupos de duplicatas exatas: **0**
- Situação: **validado estruturalmente**

### 5.6. Observação sobre imagens semelhantes

A auditoria consolidada identificou dois grupos com nomes de origem compartilhados entre diferentes conjuntos de dados.

As verificações anteriores não identificaram duplicatas binariamente idênticas entre esses conjuntos.

Entretanto, imagens com transformações, recortes ou pequenas alterações podem não apresentar hashes iguais.

Por esse motivo, recomenda-se uma análise complementar de similaridade antes do treinamento definitivo.

---

## 6. Dataset de vergalhões — Rebar Counting V1

### 6.1. Identificação

- **Categoria:** Vergalhões
- **Identificador:** `vergalhoes/rebar_counting_v1`
- **Fonte:** Roboflow Universe
- **Link:** https://universe.roboflow.com/picking-recognition-wd/rebar_counting_v1
- **Versão utilizada:** `original.zip`
- **Total de imagens:** 3
- **Total de objetos anotados:** 26
- **Formato:** YOLO Detection

### 6.2. Classes

| ID | Classe |
|---|---|
| 0 | `rebar edge` — Extremidade de vergalhão |

A classe identifica a extremidade do vergalhão, não necessariamente a barra em toda a sua extensão.

### 6.3. Distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 3 |
| Validação (`valid`) | 0 |
| Teste (`test`) | 0 |
| **Total** | **3** |

### 6.4. Resultado da validação

- Imagens processadas: **3**
- Anotações válidas: **26**
- Problemas estruturais restantes: **0**
- Grupos de duplicatas exatas: **0**
- Situação: **exploratório, sem validação ou teste**

### 6.5. Limitações

Embora as anotações tenham passado na validação estrutural, o dataset possui uma quantidade insuficiente de imagens para treinamento e avaliação confiáveis.

Os principais problemas são:

- Quantidade extremamente reduzida de imagens;
- Ausência de conjuntos de validação e teste;
- Baixa diversidade visual disponível;
- Impossibilidade de avaliar adequadamente a capacidade de generalização de um modelo.

Esse dataset será mantido para experimentação inicial e deverá ser ampliado antes de sua utilização em um treinamento definitivo.

---

## 7. Organização dos arquivos

Os datasets são armazenados no Google Drive, utilizando a seguinte estrutura:

```text
PDI_Contagem_Objetos/
│
├── datasets/
│   ├── parafusos/
│   │   └── roboflow_screws/
│   │       ├── original.zip
│   │       └── revisado.zip
│   │
│   ├── porcas/
│   │   └── nutsnbolts/
│   │       ├── original.zip
│   │       └── filtrado.zip
│   │
│   ├── arruelas/
│   │   └── washers_hardee/
│   │       ├── original.zip
│   │       ├── revisado.zip
│   │       └── deduplicado.zip
│   │
│   └── vergalhoes/
│       └── rebar_counting_v1/
│           └── original.zip
│
└── resultados/
    └── fase1_multidataset/
        ├── validacao_final/
        └── <ident
