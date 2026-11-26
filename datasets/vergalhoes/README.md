# Dataset de Vergalhões — Rebar Counting V1

## 1. Descrição

Este diretório contém o dataset **Rebar Counting V1**, utilizado no projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais**.

O dataset apresenta imagens de vergalhões, com anotações destinadas à identificação de suas extremidades.

O objetivo de sua utilização é investigar a possibilidade de identificar e contabilizar vergalhões em imagens industriais, especialmente em situações nas quais as barras são observadas pelas extremidades.

Durante a **Fase 1 — Organização, Validação e Ground Truth**, o conjunto passou por verificações estruturais das anotações e da integridade dos arquivos.

Embora tenha sido aprovado nessas verificações, sua quantidade reduzida de imagens impede uma avaliação estatisticamente confiável de modelos de reconhecimento.

Por esse motivo, o dataset foi classificado como **exploratório**.

## 2. Fonte dos dados

- **Dataset:** Rebar Counting V1
- **Plataforma:** Roboflow Universe
- **Fonte:** https://universe.roboflow.com/picking-recognition-wd/rebar_counting_v1
- **Identificador no projeto:** `vergalhoes/rebar_counting_v1`
- **Formato das anotações:** YOLO Detection
- **Quantidade de classes:** 1

As condições de licença e redistribuição devem ser verificadas na página original do dataset.

## 3. Arquivos disponíveis

```text
datasets/
└── vergalhoes/
    ├── README.md
    └── rebar_counting_v1/
        └── original.zip
```

| Arquivo | Descrição |
|---|---|
| `original.zip` | Versão original do dataset, preservada sem modificações |

**Versão selecionada para utilização exploratória:** `rebar_counting_v1/original.zip`.

Nenhuma versão corrigida ou filtrada foi necessária durante a validação estrutural.

## 4. Classe identificada

O dataset possui uma classe.

| ID | Classe | Objetos anotados |
|---|---|---:|
| 0 | `rebar edge` — Extremidade de vergalhão | 26 |
| **Total** | | **26** |

A classe `rebar edge` identifica a extremidade de um vergalhão, não necessariamente a barra em toda a sua extensão.

Essa distinção é importante para a definição da futura estratégia de contagem.

Em imagens que apresentam as extremidades de um conjunto de barras, a contagem das extremidades corretamente identificadas pode servir como aproximação da quantidade de vergalhões.

Entretanto, essa relação depende das condições de captura e da visibilidade dos objetos.

## 5. Quantidade e distribuição das imagens

O dataset possui três imagens.

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 3 |
| Validação (`valid`) | 0 |
| Teste (`test`) | 0 |
| **Total** | **3** |

### Estatísticas

- **3 imagens;**
- **26 objetos anotados;**
- **1 classe;**
- **0 problemas estruturais identificados;**
- **0 grupos de duplicatas exatas.**

Todas as imagens pertencem ao conjunto de treinamento.

## 6. Auditoria e validação estrutural

Durante a Fase 1, foram realizadas verificações para identificar problemas que pudessem comprometer a leitura das imagens e das anotações.

As verificações incluíram:

1. Integridade dos arquivos.
2. Existência das imagens.
3. Correspondência entre imagens e arquivos de anotação.
4. Compatibilidade dos identificadores de classe com o `data.yaml`.
5. Validação do formato YOLO Detection.
6. Verificação das coordenadas normalizadas.
7. Identificação de caixas delimitadoras inválidas.
8. Identificação de anotações órfãs.
9. Detecção de duplicatas exatas utilizando SHA-256.
10. Conferência da quantidade de imagens e objetos anotados.

### Resultado da validação

| Critério | Resultado |
|---|---|
| Imagens esperadas | 3 — Aprovado |
| Objetos anotados | 26 — Aprovado |
| Problemas estruturais | 0 — Aprovado |
| Duplicatas exatas | 0 — Aprovado |
| Conjunto de treinamento | Presente |
| Conjunto de validação | Ausente |
| Conjunto de teste | Ausente |
| **Status** | **Exploratório, sem validação ou teste** |

Os resultados confirmaram a integridade estrutural das anotações existentes.

Entretanto, a aprovação estrutural não significa que o conjunto seja suficiente para treinamento e avaliação de modelos.

## 7. Limitações identificadas

### 7.1. Quantidade insuficiente de imagens

O dataset possui apenas três imagens.

Essa quantidade é insuficiente para representar adequadamente as diferentes condições visuais que podem ocorrer em ambientes industriais.

Entre essas condições estão:

- Diferentes quant
