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
- Verificação dos limites de caixas delimitadoras;
- Verificação da estrutura de anotações de segmentação;
- Correspondência entre arquivos de imagens e anotações;
- Detecção de duplicatas exatas.

### 6.1. Problemas identificados

A auditoria identificou inconsistências em determinadas anotações de detecção.

Foram necessárias as seguintes intervenções:

| Tratamento | Quantidade |
|---|---:|
| Caixas delimitadoras ajustadas aos limites da imagem | 2 |
| Anotações inválidas descartadas | 3 |

### 6.2. Correções realizadas

As duas caixas delimitadoras foram ajustadas para permanecer dentro dos limites permitidos da imagem.

As três anotações consideradas inválidas foram removidas da versão derivada.

As demais anotações válidas foram preservadas.

Após esse processo, foi gerado o arquivo `revisado.zip`, mantendo o `original.zip` disponível para auditoria.

## 7. Validação estrutural

A versão revisada foi novamente analisada pelo notebook consolidado da Fase 1.

As verificações incluíram:

1. Existência dos arquivos de imagem e anotação.
2. Correspondência entre imagens e rótulos.
3. Compatibilidade dos identificadores de classe com o `data.yaml`.
4. Estrutura das anotações YOLO Detection.
5. Estrutura das anotações YOLO Segmentation.
6. Validade geométrica das coordenadas.
7. Identificação de anotações inválidas ou órfãs.
8. Verificação de duplicatas binariamente idênticas por SHA-256.
9. Conferência da distribuição entre treinamento, validação e teste.
10. Verificação dos totais esperados de imagens e objetos.

### Resultado final

| Critério | Resultado |
|---|---|
| Quantidade de imagens | Aprovado |
| Quantidade de objetos | Aprovado |
| Integridade estrutural das anotações | Aprovado |
| Ausência de duplicatas exatas | Aprovado |
| Conjuntos de treinamento, validação e teste | Presentes |
| **Status** | **Validado estruturalmente** |

Nenhum problema estrutural ou grupo de duplicatas exatas foi identificado na versão revisada durante a execução consolidada.

## 8. Ground Truth

O Ground Truth foi construído a partir das anotações válidas presentes no dataset.

Cada linha válida de anotação corresponde a uma instância contabilizada.

Para cada imagem, foram registrados:

- Identificador do dataset;
- Caminho da imagem;
- Conjunto de origem;
- Quantidade total de objetos anotados;
- Quantidade de objetos por classe;
- Existência de problemas estruturais;
- Necessidade de revisão.

O Ground Truth é gerado pelo notebook:

`notebooks/fase1_multidataset_consolidada.ipynb`

Os relatórios também estão disponíveis na pasta de resultados da Fase 1 no Google Drive.

**Importante:** o Ground Truth foi derivado dos rótulos existentes. A validação estrutural não comprova que todos os objetos visíveis nas imagens estejam anotados.

## 9. Limitações e cuidados

Apesar de ter passado na validação estrutural, o dataset possui características que devem ser consideradas nas próximas fases.

### 9.1. Diferentes formatos de anotação

O conjunto apresenta anotações nos formatos YOLO Detection e YOLO Segmentation.

Na base final de contagem, `scripts/preparar_dados.py` trata os dois formatos: os polígonos viram a caixa que os envolve.

Anotações de segmentação não devem ser interpretadas diretamente como caixas delimitadoras sem a conversão apropriada.

### 9.2. Classes específicas

As quatro classes representam diferentes tipos de parafusos.

Na base final de contagem, as quatro classes foram agrupadas na classe `parafuso`, no mapeamento de `scripts/preparar_dados.py`.

### 9.3. Qualidade visual das anotações

A auditoria realizada verificou a integridade e a geometria dos rótulos.

Uma inspeção visual complementar poderá ser necessária para avaliar a correspondência entre os objetos e as regiões anotadas.

### 9.4. Diversidade das imagens

A quantidade de imagens disponível não garante, isoladamente, diversidade suficiente de cenários, iluminação, orientação e sobreposição de objetos.

O desempenho dos modelos deverá ser avaliado nos conjuntos de validação e teste.

## 10. Estado atual

**Status: validado estruturalmente na Fase 1.**

A versão `revisado.zip` atende aos critérios de:

- Integridade dos arquivos;
- Validade estrutural das anotações;
- Quantidades esperadas de imagens e objetos;
- Ausência de duplicatas binárias exatas.

O dataset está organizado e disponível no repositório para utilização nas próximas etapas, observadas as condições de licença da fonte.

O dataset foi usado na base final de contagem (seção 11), na pipeline clássica e no treino do detector YOLO de parafuso.

## 11. Uso na base final de contagem

`scripts/preparar_dados.py` lê `roboflow_screws/revisado.zip` e usa todas as classes como `parafuso`. O script exclui por regra as 3 imagens cujo nome contém `Parafusos_Dourados_0089`. O repositório não registra o motivo dessa exclusão. Das 460 imagens restantes, 385 entram na base final (335 em `train`, 25 em `valid` e 25 em `test`). As outras 75 foram omitidas: em `valid` e `test` o script mantém só a primeira imagem de cada grupo de imagens parecidas.

A base final tem 2.715 objetos deste dataset, dos quais 2.677 vêm de polígonos e 38 de caixas. Os números vêm de `data_final/manifesto.json`, gerado pelo script.

---

**Projeto:** Contagem de Objetos Industriais  
**Disciplina:** Processamento Digital de Imagens  
**Etapa:** Fase 1 — Organização, Validação e Ground Truth  
**Dataset:** Roboflow Screws  
**Versão aprovada:** `revisado.zip`
