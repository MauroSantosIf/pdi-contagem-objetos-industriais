# Contagem de Objetos Industriais — Processamento Digital de Imagens

Projeto acadêmico da disciplina de **Processamento Digital de Imagens (PDI)**, desenvolvido para identificação e contagem automática de objetos industriais, utilizando Python, OpenCV e técnicas de visão computacional.

O projeto investiga diferentes métodos de preparação, processamento e reconhecimento de imagens, com o objetivo de desenvolver uma solução capaz de identificar e contabilizar objetos presentes em ambientes industriais.

## 1. Objetivo do projeto

O objetivo principal é desenvolver e avaliar uma solução computacional para identificar, classificar e contar objetos industriais em imagens.

Inicialmente, o projeto contempla quatro categorias:

- **Parafusos:** identificação de diferentes tipos de parafusos;
- **Porcas:** identificação e contagem de porcas;
- **Arruelas:** identificação e contagem de arruelas;
- **Vergalhões:** investigação de técnicas de contagem por identificação de extremidades.

A proposta envolve desde a preparação dos dados até a avaliação dos resultados obtidos por técnicas de processamento digital de imagens e visão computacional.

## 2. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento dos algoritmos |
| OpenCV | Processamento digital de imagens |
| NumPy | Manipulação de dados e matrizes |
| Pandas | Organização e análise dos resultados |
| Matplotlib | Visualização de imagens e gráficos |
| PyYAML | Leitura das configurações dos datasets |
| Google Colab | Ambiente de execução dos notebooks |
| Git e GitHub | Versionamento e colaboração |

As tecnologias poderão ser ampliadas conforme a evolução do projeto.

## 3. Datasets utilizados

Foram selecionados quatro datasets com imagens de objetos industriais.

Durante a Fase 1, esses conjuntos foram organizados, auditados e validados estruturalmente.

| Dataset | Categoria | Imagens | Objetos anotados | Versão selecionada |
|---|---|---:|---:|---|
| Roboflow Screws | Parafusos | 463 | 3.230 | `revisado.zip` |
| NutsNBolts | Porcas, parafusos e arruelas | 107 | 2.832 | `filtrado.zip` |
| Washers Hardee | Arruelas | 402 | 3.949 | `deduplicado.zip` |
| Rebar Counting V1 | Vergalhões | 3 | 26 | `original.zip` |
| **Total** | | **975** | **10.037** | |

Os valores representam as quantidades de instâncias anotadas nos datasets, conforme os rótulos existentes.

### Documentação individual

- [Dataset de parafusos](datasets/parafusos/README.md)
- [Dataset NutsNBolts](datasets/porcas/README.md)
- [Dataset de arruelas](datasets/arruelas/README.md)
- [Dataset de vergalhões](datasets/vergalhoes/README.md)

Os datasets estão organizados em diretórios específicos, com preservação das versões originais e das versões tratadas durante a Fase 1.

A utilização e redistribuição dos arquivos devem respeitar as licenças das respectivas fontes.

## 4. Estrutura do repositório

A estrutura principal está organizada da seguinte forma:

```text
pdi-contagem-objetos-industriais/
│
├── README.md
├── .gitignore
│
├── datasets/
│   ├── parafusos/
│   │   ├── README.md
│   │   └── roboflow_screws/
│   │       ├── original.zip
│   │       └── revisado.zip
│   │
│   ├── porcas/
│   │   ├── README.md
│   │   └── nutsnbolts/
│   │       ├── original.zip
│   │       └── filtrado.zip
│   │
│   ├── arruelas/
│   │   ├── README.md
│   │   └── washers_hardee/
│   │       ├── original.zip
│   │       ├── revisado.zip
│   │       └── deduplicado.zip
│   │
│   └── vergalhoes/
│       ├── README.md
│       └── rebar_counting_v1/
│           └── original.zip
│
├── notebooks/
│   └── fase1_dataset_ground_truth.ipynb
│
└── docs/
    ├── datasets.md
    └── fase1_metodologia.md
```

A estrutura poderá ser ampliada conforme a implementação das próximas fases.

## 5. Etapas do projeto

O desenvolvimento foi dividido em etapas para facilitar a organização das atividades e a validação dos resultados.

### Fase 1 — Organização, Validação e Ground Truth

**Status: concluída dentro do escopo de preparação e validação estrutural.**

**Responsável:** Vivian Beatriz.

Esta fase compreendeu:

1. Levantamento e organização dos datasets.
2. Identificação das classes e dos formatos de anotação.
3. Validação estrutural dos rótulos.
4. Auditoria de imagens duplicadas.
5. Correção e filtragem de inconsistências.
6. Preservação dos dados originais.
7. Geração do Ground Truth.
8. Consolidação dos resultados e das limitações.

#### Resultados

| Indicador | Resultado |
|---|---:|
| Datasets analisados | 4 |
| Imagens processadas | 975 |
| Instâncias anotadas | 10.037 |
| Problemas estruturais restantes | 0 |
| Grupos de duplicatas exatas restantes | 0 |

Os quatro datasets passaram nos critérios de integridade estrutural e ausência de duplicatas binárias.

Entretanto, o dataset de vergalhões possui somente três imagens, todas no conjunto de treinamento. Por isso, permanece classificado como exploratório.

Também foram identificados dois grupos com nomes de origem compartilhados entre diferentes conjuntos no dataset de arruelas. Isso exige cautela na investigação de imagens visualmente relacionadas antes do treinamento.

**Notebook:** [Fase 1 — Organização e Ground Truth](notebooks/fase1_dataset_ground_truth.ipynb)

**Documentação:** [Metodologia da Fase 1](docs/fase1_metodologia.md)

### Fase 2 — Pré-processamento de Imagens

**Status: planejada.**

Esta fase terá como objetivo investigar e aplicar técnicas para preparar as imagens antes do reconhecimento dos objetos.

Entre as técnicas previstas estão:

- Conversão para escala de cinza;
- Redução de ruído por filtros;
- Correção de iluminação;
- Equalização de histograma;
- Comparação entre imagens originais e processadas.

As técnicas serão avaliadas para determinar quais são adequadas às características dos datasets e à estratégia de reconhecimento adotada.

### Fase 3 — Reconhecimento e Contagem

**Status: planejada.**

Esta fase investigará técnicas para localizar, identificar e contabilizar os objetos industriais presentes nas imagens.

A abordagem de reconhecimento será definida conforme os experimentos e as necessidades do projeto.

Poderão ser consideradas técnicas clássicas de visão computacional e modelos de detecção de objetos.

### Fase 4 — Avaliação dos Resultados

**Status: planejada.**

A etapa de avaliação deverá comparar as contagens produzidas pelo sistema com as referências do Ground Truth.

Poderão ser utilizadas métricas de erro de contagem e medidas de desempenho relacionadas à detecção e à classificação.

A escolha das métricas dependerá da abordagem efetivamente implementada.

## 6. Como executar a Fase 1

O notebook da Fase 1 foi desenvolvido para execução no Google Colab.

### Requisitos

- Conta Google para utilizar o Colab;
- Python e bibliotecas utilizadas no notebook;
- Acesso aos arquivos ZIP dos datasets.

### Procedimento

1. Acesse o notebook [Fase 1 — Organização e Ground Truth](notebooks/fase1_dataset_ground_truth.ipynb).
2. Abra o arquivo no Google Colab.
3. Organize os datasets na pasta esperada do Google Drive.
4. Autorize o acesso ao Drive quando solicitado.
5. Execute as células na ordem apresentada.
6. Confira os relatórios consolidados e os critérios de aceitação.

O notebook utiliza, por padrão, a seguinte pasta:

```text
/content/drive/MyDrive/PDI_Contagem_Objetos/
```

Dentro dela, os arquivos ZIP devem estar organizados na pasta `datasets/`, respeitando a estrutura indicada neste README.

**Observação:** embora os datasets também estejam disponíveis no repositório, a versão atual do notebook utiliza caminhos do Google Drive. A execução diretamente a partir do clone do GitHub requer adaptação dos caminhos de entrada.

### Resultados gerados

A execução produz arquivos como:

- `ground_truth.csv`;
- `problemas_anotacoes.csv`;
- `duplicatas_exatas.csv`;
- `suspeitas_entre_splits.csv`;
- `resumo.json`;
- `consolidado.csv`;
- `criterios_aceitacao.csv`;
- `manifesto_execucao.json`.

Os relatórios são armazenados em uma pasta identificada pela data e pelo horário da execução.

## 7. Ground Truth e qualidade dos dados

O Ground Truth corresponde às referências utilizadas para contabilizar os objetos em cada imagem.

Na Fase 1, os valores foram derivados das anotações fornecidas pelos datasets.

As verificações realizadas abrangeram:

- Validade dos identificadores de classe;
- Estrutura das anotações YOLO Detection e Segmentation;
- Limites geométricos das coordenadas;
- Correspondência entre imagens e rótulos;
- Identificação de anotações inválidas;
- Detecção de duplicatas exatas com SHA-256.

A aprovação estrutural não garante que todos os objetos visíveis tenham sido anotados corretamente.

Antes do treinamento, ainda deverão ser consideradas a qualidade visual das anotações, a semelhança entre imagens e a padronização das classes.

## 8. Documentação

Os documentos do projeto estão organizados em `docs/`.

- [Inventário dos datasets](docs/datasets.md)
- [Metodologia da Fase 1](docs/fase1_metodologia.md)

Cada dataset possui também um README próprio, contendo informações sobre sua origem, versões, classes, auditorias e limitações.

## 9. Limitações atuais

As principais limitações identificadas até o momento são:

1. O conjunto de vergalhões possui apenas três imagens, sem validação ou teste.
2. O dataset NutsNBolts possui somente cinco imagens no conjunto de validação e cinco no conjunto de teste.
3. Os datasets apresentam classes e formatos de anotação diferentes, que ainda precisam ser padronizados para um treinamento conjunto.
4. A validação estrutural não substitui uma inspeção visual completa.
5. A identificação por SHA-256 não detecta necessariamente imagens visualmente semelhantes.
6. As licenças de utilização e redistribuição devem ser verificadas nas fontes originais.

Essas limitações serão consideradas durante a implementação das próximas etapas.

## 10. Situação atual

| Etapa | Situação |
|---|---|
| Organização dos datasets | Concluída |
| Auditoria e validação estrutural | Concluída |
| Geração do Ground Truth | Concluída |
| Documentação individual dos datasets | Concluída |
| Pré-processamento de imagens | Planejado |
| Reconhecimento e contagem | Planejado |
| Avaliação de desempenho | Planejada |

O projeto encontra-se com a **Fase 1 concluída e documentada**, dispondo de quatro datasets organizados e dos respectivos resultados de validação estrutural.

A próxima etapa será a implementação e avaliação das técnicas de pré-processamento.

---

**Projeto:** Contagem de Objetos Industriais  
**Disciplina:** Processamento Digital de Imagens  
**Repositório:** https://github.com/MauroSantosIf/pdi-contagem-objetos-industriais
