# Dataset de Arruelas — Washers Hardee

## 1. Descrição

Este diretório contém o dataset **Washers Hardee**, utilizado no projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais**.

O dataset é composto por imagens de arruelas industriais, com anotações que permitem identificar e contabilizar os objetos presentes nas imagens.

Durante a **Fase 1 — Organização, Validação e Ground Truth**, o conjunto passou por uma auditoria estrutural e um processo de deduplicação para remover imagens redundantes e melhorar a consistência dos dados.

O objetivo é disponibilizar uma versão organizada e validada estruturalmente para as próximas etapas do projeto.

## 2. Fonte dos dados

- **Dataset:** Washers Hardee
- **Plataforma:** Roboflow
- **Identificador no projeto:** `arruelas/washers_hardee`
- **Formato das anotações:** YOLO Detection
- **Quantidade de classes:** 1

O link específico do dataset original e suas condições de licença e redistribuição devem ser confirmados na plataforma de origem.

## 3. Arquivos disponíveis

```text
datasets/
└── arruelas/
    ├── README.md
    └── washers_hardee/
        ├── original.zip
        ├── revisado.zip
        └── deduplicado.zip
```

| Arquivo | Descrição |
|---|---|
| `original.zip` | Dataset original, preservado sem alterações |
| `revisado.zip` | Versão intermediária após a remoção inicial de imagens redundantes |
| `deduplicado.zip` | Versão final após a deduplicação completa |

**Versão selecionada para utilização no projeto:** `washers_hardee/deduplicado.zip`.

Os arquivos anteriores foram preservados para permitir auditoria e rastreabilidade.

## 4. Classes identificadas

O dataset possui apenas uma classe.

| ID | Classe | Objetos anotados |
|---|---|---:|
| 0 | `washer` — Arruela | 3.949 |
| **Total** | | **3.949** |

Os identificadores das classes são específicos deste dataset.

Em um treinamento conjunto com outros datasets, será necessário padronizar os identificadores e a nomenclatura das classes equivalentes.

## 5. Quantidade e distribuição das imagens

A versão final contém 402 imagens.

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 280 |
| Validação (`valid`) | 80 |
| Teste (`test`) | 42 |
| **Total** | **402** |

### Estatísticas da versão final

- **402 imagens;**
- **3.949 objetos anotados;**
- **1 classe;**
- **0 problemas estruturais;**
- **0 grupos de duplicatas exatas.**

O conjunto mantém imagens distribuídas entre treinamento, validação e teste.

## 6. Auditoria e tratamento dos dados

A auditoria foi realizada em duas etapas, preservando o arquivo original.

### 6.1. Dataset original

A versão original apresentava:

| Indicador | Quantidade |
|---|---:|
| Imagens | 736 |
| Objetos anotados | 7.507 |
| Problemas estruturais | 0 |

Embora as anotações estivessem estruturalmente válidas, foram identificadas imagens redundantes.

### 6.2. Primeira revisão

Na primeira etapa, foram removidas três cópias redundantes.

O arquivo resultante foi denominado `revisado.zip`.

| Indicador | Quantidade |
|---|---:|
| Imagens | 733 |
| Objetos anotados | 7.430 |

Essa versão intermediária foi preservada para registrar o histórico do tratamento.

### 6.3. Deduplicação completa

Uma segunda auditoria foi realizada para identificar imagens binariamente idênticas.

O processo utilizou hashes SHA-256, calculados sobre o conteúdo das imagens.

Foram identificados:

| Indicador | Resultado |
|---|---:|
| Grupos de imagens duplicadas | 245 |
| Cópias excedentes | 331 |
| Grupos com anotações idênticas | 245 |
| Grupos com anotações diferentes | 0 |
| Duplicatas exatas entre conjuntos distintos | 0 |

Também foram calculados hashes das anotações correspondentes.

Essa verificação confirmou que as imagens duplicadas possuíam arquivos de anotação idênticos dentro dos respectivos grupos.

Assim, foi possível remover as cópias redundantes mantendo uma imagem e sua anotação em cada grupo.

### 6.4. Resultado da deduplicação

| Versão | Imagens | Objetos anotados |
|---|---:|---:|
| `original.zip` | 736 | 7.507 |
| `revisado.zip` | 733 | 7.430 |
| `deduplicado.zip` | 402 | 3.949 |

Ao todo, foram eliminadas **334 cópias redundantes de imagens**, considerando as duas etapas.

A redução das anotações corresponde à remoção dos rótulos associados às imagens duplicadas, e não à exclusão de novas classes ou de objetos distintos identificados durante a auditoria de hashes.

A versão final foi salva em `deduplicado.zip`.

## 7. Validação estrutural

Após a deduplicação, foi executada uma nova validação para confirmar a integridade do dataset.

As verificações incluíram:

1. Integridade do arquivo ZIP.
2. Existência dos arquivos de imagem e anotação.
3
