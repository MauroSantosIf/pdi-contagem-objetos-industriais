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

- Diferentes quantidades de vergalhões;
- Variações de iluminação;
- Mudanças de perspectiva;
- Sobreposição de objetos;
- Alterações de escala;
- Diferentes cenários e fundos.

A pequena quantidade de exemplos limita a capacidade de avaliar a generalização de um modelo.

### 7.2. Ausência de validação e teste

As três imagens disponíveis pertencem exclusivamente ao conjunto `train`.

Não existem imagens nos conjuntos `valid` e `test`.

Por esse motivo, não é possível realizar uma avaliação independente e confiável de desempenho utilizando somente esse dataset.

### 7.3. Limitação da classe anotada

O conjunto utiliza a classe `rebar edge`, correspondente à extremidade do vergalhão.

Essa abordagem pode ser adequada para imagens de feixes de barras observados pelas extremidades, mas não representa necessariamente a identificação de um vergalhão completo.

A estratégia de contagem deverá considerar essa característica.

### 7.4. Qualidade visual das anotações

A validação realizada foi estrutural.

Ela não garante que todas as extremidades visíveis estejam corretamente anotadas.

Uma inspeção visual complementar será necessária caso o conjunto seja utilizado em experimentos de reconhecimento e contagem.

## 8. Decisão técnica da equipe

Durante a Fase 1, foi decidido **preservar o dataset original sem alterações**.

Também foi decidido não dividir artificialmente as três imagens entre treinamento, validação e teste, pois essa divisão produziria conjuntos extremamente pequenos e sem representatividade suficiente.

O dataset permanece disponível para:

- Análises exploratórias;
- Inspeção de anotações;
- Testes iniciais de leitura e processamento de imagens;
- Investigação de estratégias de contagem por extremidades;
- Comparação com novos datasets de vergalhões.

A utilização para treinamento definitivo dependerá da obtenção de novas imagens representativas.

## 9. Ground Truth

O Ground Truth foi construído a partir das 26 anotações válidas existentes no dataset.

Para cada imagem, o notebook registra:

- Identificador do dataset;
- Caminho da imagem;
- Conjunto de origem;
- Quantidade de extremidades anotadas;
- Existência de inconsistências estruturais;
- Necessidade de revisão.

O Ground Truth é gerado pelo notebook:

`notebooks/fase1_multidataset_consolidada.ipynb`

Os relatórios da validação são armazenados no diretório de resultados da Fase 1.

**Importante:** as 26 instâncias representam extremidades anotadas. Sua interpretação como quantidade de barras depende das condições das imagens.

## 10. Recomendações para ampliação

Antes de utilizar o dataset para treinamento e avaliação de um modelo de contagem, recomenda-se:

1. Buscar novos datasets públicos com imagens de vergalhões.
2. Verificar as licenças e as condições de utilização.
3. Coletar imagens com maior diversidade de perspectivas e iluminação.
4. Priorizar imagens nítidas e com anotações consistentes.
5. Definir se a contagem será realizada por extremidades ou pela identificação da barra completa.
6. Padronizar as classes e os formatos de anotação.
7. Criar conjuntos independentes de treinamento, validação e teste.
8. Reexecutar a auditoria e a validação estrutural da Fase 1.

A ampliação deverá ser documentada, preservando a versão original utilizada nesta etapa.

## 11. Estado atual

**Status: exploratório, sem conjuntos de validação e teste.**

O dataset apresentou:

- 3 imagens;
- 26 extremidades anotadas;
- Nenhum problema estrutural identificado;
- Nenhuma duplicata exata;
- Todas as imagens no conjunto de treinamento.

A versão original foi preservada para rastreabilidade.

O conjunto está disponível para experimentação, mas não é considerado suficiente para treinamento e avaliação confiáveis de um modelo de reconhecimento de vergalhões.

## Situação na base final de contagem

Este dataset não entra em `data_final/` nem nas comparações entre a pipeline clássica e o detector YOLO. A base final usa `parafuso`, `porca` e `arruela`, e o dataset tem só 3 imagens, sem `valid` nem `test`.

---

**Projeto:** Contagem de Objetos Industriais  
**Disciplina:** Processamento Digital de Imagens  
**Etapa:** Fase 1 — Organização, Validação e Ground Truth  
**Dataset:** Rebar Counting V1  
**Versão selecionada:** `original.zip`  
**Classificação:** Exploratório
