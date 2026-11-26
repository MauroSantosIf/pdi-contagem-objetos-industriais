# Dataset de Parafusos — Roboflow Screws

## 1. Descrição

Este diretório contém o dataset **Roboflow Screws**, utilizado no projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais**.

O dataset reúne imagens de diferentes tipos de parafusos industriais, organizados em quatro classes.

Seu objetivo no projeto é fornecer imagens e anotações para as etapas de preparação, reconhecimento, classificação e contagem de parafusos.

Durante a **Fase 1 — Organização, Validação e Ground Truth**, o dataset passou por uma auditoria estrutural das anotações, correção de inconsistências e geração de uma versão revisada.

## 2. Fonte dos dados

- **Dataset:** Roboflow Screws
- **Plataforma:** Roboflow
- **Identificador no projeto:** `parafusos/roboflow_screws`
- **Formatos encontrados:** YOLO Detection e YOLO Segmentation
- **Quantidade de classes:** 4

O link específico do projeto de origem e sua licença de utilização devem ser registrados após conferência na plataforma Roboflow.

A disponibilização dos arquivos neste repositório deve respeitar as condições de redistribuição estabelecidas pela fonte.

## 3. Arquivos disponíveis

```text
datasets/
└── parafusos/
    ├── README.md
    └── roboflow_screws/
        ├── original.zip
        └── revisado.zip
```

| Arquivo | Descrição |
|---|---|
| `original.zip` | Dataset original, preservado para rastreabilidade |
| `revisado.zip` | Versão com as correções de anotações realizadas na Fase 1 |

**Versão aprovada para utilização no projeto:** `roboflow_screws/revisado.zip`.

O arquivo original foi preservado e não sofreu modificações durante a revisão.

## 4. Classes identificadas

O dataset contém quatro classes de parafusos.

| ID | Classe | Objetos anotados |
|---|---|---:|
| 0 | Parafuso D2 4.0x25 C4 | 844 |
| 1 | Parafuso D3 35x30 c6 | 805 |
| 2 | Parafuso D4 4.5x16 C16 | 760 |
| 3 | parafuso D1 35x12 c2 | 821 |
| **Total** | | **3.230** |

Os nomes das classes foram mantidos conforme a configuração do dataset original.

Os identificadores são específicos deste conjunto e deverão ser mapeados para uma nomenclatura comum caso os datasets sejam integrados em um treinamento conjunto.

## 5. Quantidade e distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 387 |
| Validação (`valid`) | 36 |
| Teste (`test`) | 40 |
| **Total** | **463** |

A versão revisada contém:

- **463 imagens;**
- **3.230 objetos anotados;**
- **4 classes;**
- **0 problemas estruturais restantes;**
- **0 grupos de duplicatas exatas.**

As imagens estão distribuídas entre treinamento, validação e teste, seguindo a organização encontrada no dataset.

## 6. Auditoria e tratamento dos dados

Durante a Fase 1, foram realizadas verificações das anotações YOLO para identificar inconsistências que pudessem comprometer a utilização dos dados.

Entre as verificações executadas estavam:

- Identificação das classes declaradas no `data.yaml`;
- Validação de coordenadas normalizadas;
- Verificação
