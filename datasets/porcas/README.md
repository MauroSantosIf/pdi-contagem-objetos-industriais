# Dataset de Porcas — NutsNBolts

## 1. Descrição

Este diretório contém o dataset **NutsNBolts**, utilizado no projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais**.

Embora esteja organizado na categoria `porcas`, o dataset contém imagens e anotações de três tipos de objetos industriais:

- **Bolt:** parafusos;
- **Nut:** porcas;
- **Washer:** arruelas.

O objetivo é utilizar essas imagens nas etapas de preparação, reconhecimento e contagem de objetos industriais.

O dataset passou por auditoria e validação estrutural durante a **Fase 1 — Organização, Validação e Ground Truth**.

## 2. Fonte dos dados

- **Dataset:** NutsNBolts
- **Plataforma:** Roboflow Universe
- **Fonte:** https://universe.roboflow.com/tktk-hodc3/nutsnbolts
- **Formato das anotações:** YOLO Detection
- **Quantidade de classes:** 3

As condições de licença e redistribuição devem ser verificadas na página de origem do dataset.

## 3. Arquivos disponíveis

```text
datasets/
└── porcas/
    ├── README.md
    └── nutsnbolts/
        ├── original.zip
        └── filtrado.zip
```

| Arquivo | Descrição |
|---|---|
| `original.zip` | Versão original do dataset, preservada para rastreabilidade |
| `filtrado.zip` | Versão tratada e aprovada na validação estrutural da Fase 1 |

**Versão selecionada para utilização no projeto:** `nutsnbolts/filtrado.zip`.

O arquivo original não foi modificado durante a filtragem.

## 4. Classes identificadas

A versão filtrada apresenta a seguinte distribuição de objetos anotados:

| ID | Classe | Descrição | Objetos |
|---|---|---|---:|
| 0 | `bolt` | Parafuso | 1.564 |
| 1 | `nut` | Porca | 725 |
| 2 | `washer` | Arruela | 543 |
| **Total** | | | **2.832** |

Os IDs das classes são específicos deste dataset e não devem ser utilizados diretamente como identificadores globais em um treinamento conjunto.

## 5. Quantidade e distribuição das imagens

| Conjunto | Imagens |
|---|---:|
| Treinamento (`train`) | 97 |
| Validação (`valid`) | 5 |
| Teste (`test`) | 5 |
| **Total** | **107** |

A versão final contém:

- **107 imagens;**
- **2.832 objetos anotados;**
- **3 classes;**
- **0 problemas estruturais identificados;**
- **0 grupos de duplicatas exatas.**

## 6. Auditoria e tratamento dos dados

Durante a Fase 1, foram analisadas as anotações do dataset original.

A auditoria identificou:

| Indicador | Resultado |
|---|---:|
| Imagens originais | 130 |
| Objetos com anotações válidas na versão original | 3.408 |
| Anotações com caixas fora dos limites | 32 |
| Imagens afetadas | 23 |
| Imagens mantidas após filtragem | 107 |
| Objetos anotados na versão filtrada | 2.832 |

### Decisão técnica

A equipe optou por remover da versão derivada as 23 imagens que apresentavam problemas de anotação.

A decisão teve como objetivo evitar o uso de caixas delimitadoras inválidas nas etapas posteriores.

As imagens removidas permanecem disponíveis no arquivo `original.zip`, permitindo auditoria ou recuperação futura.

Após o tratamento, foi gerado o arquivo `filtrado.zip`.

## 7. Validação estrutural

A versão filtrada passou pelas seguintes verificações:

1. Existência dos arquivos de imagens e anotações.
2. Correspondência entre imagens e rótulos.
3. Identificação das classes declaradas no `data.yaml`.
4. Validação das coordenadas YOLO.
5. Identificação de caixas delimitadoras inválidas.
6. Detecção de arquivos de anotação órfãos.
7. Verificação de duplicatas exatas utilizando SHA-256.
8. Conferência da distribuição entre treinamento, validação e teste.
9. Conferência da quantidade de imagens e objetos anotados.

### Resultado final

| Critério | Resultado |
|---|---|
| Quantidade de imagens | Aprovado |
| Quantidade de objetos | Aprovado |
| Integridade estrutural das anotações | Aprovado |
| Ausência de duplicatas exatas | Aprovado |
| Conjuntos de treinamento, validação e teste | Presentes |
| **Status** | **Validado estruturalmente** |

A validação não constitui uma inspeção visual completa das imagens e anotações.

## 8. Ground Truth

O Ground Truth foi construído a partir das anotações válidas do dataset.

Para cada imagem, o notebook registra:

- Caminho da imagem;
- Conjunto de origem;
- Quantidade total de objetos anotados;
- Quantidade de objetos de cada classe;
- Existência de problemas estruturais;
- Necessidade de revisão.

Os relatórios são gerados pelo notebook da Fase 1:

`notebooks/fase1_dataset_ground_truth.ipynb`

Os resultados consolidados encontram-se na pasta de resultados da Fase 1, no Google Drive.

## 9. Limitações

Embora o dataset tenha sido validado estruturalmente, apresenta limitações importantes.

### 9.1. Quantidade reduzida de imagens

A versão filtrada contém 107 imagens, com apenas cinco imagens para validação e cinco para teste.

Esses conjuntos são pequenos, o que pode resultar em avaliações com elevada variabilidade.

### 9.2. Redução do dataset original

A filtragem removeu 23 imagens que apresentavam anotações inválidas.

Essa decisão resolveu as inconsistências estruturais detectadas, mas reduziu a quantidade e possivelmente a diversidade dos dados disponíveis.

### 9.3. Qualidade visual das anotações

A validação estrutural não garante que todos os objetos visíveis estejam anotados corretamente.

Uma inspe
