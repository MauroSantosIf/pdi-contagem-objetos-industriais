# Fase 2 — Metodologia de Pré-processamento de Imagens

## 1. Introdução

A segunda fase do projeto acadêmico de **Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais** tem como objetivo preparar as imagens para as etapas posteriores de reconhecimento e contagem de objetos.

Após a organização e validação dos datasets na Fase 1, serão aplicadas e avaliadas técnicas de processamento digital para reduzir ruídos, padronizar características visuais e investigar possíveis melhorias na identificação dos objetos.

O pré-processamento será implementado em Python, utilizando principalmente a biblioteca OpenCV, com execução no Google Colab.

Esta documentação apresenta a metodologia definida para a Fase 2. Os resultados experimentais e a escolha definitiva dos parâmetros serão registrados após a execução e avaliação.

---

## 2. Objetivos

### 2.1. Objetivo geral

Desenvolver e avaliar um pipeline de pré-processamento para imagens de objetos industriais, preservando suas características relevantes para reconhecimento e contagem.

### 2.2. Objetivos específicos

- Realizar a leitura das imagens dos datasets validados na Fase 1.
- Converter imagens para escala de cinza.
- Avaliar técnicas de redução de ruído.
- Investigar métodos de correção de iluminação e melhoria de contraste.
- Comparar visualmente as imagens originais e processadas.
- Definir um pipeline reproduzível.
- Preservar a correspondência entre imagens e anotações.
- Gerar versões processadas sem alterar os datasets originais.

---

## 3. Datasets utilizados

Serão utilizadas as versões aprovadas na Fase 1.

| Dataset | Versão | Imagens | Objetos anotados |
|---|---|---:|---:|
| Roboflow Screws | `revisado.zip` | 463 | 3.230 |
| NutsNBolts | `filtrado.zip` | 107 | 2.832 |
| Washers Hardee | `deduplicado.zip` | 402 | 3.949 |
| Rebar Counting V1 | `original.zip` | 3 | 26 |
| **Total** | | **975** | **10.037** |

O dataset de vergalhões será utilizado apenas para análise exploratória, pois apresenta somente três imagens de treinamento, sem conjuntos de validação e teste.

As versões aprovadas não serão modificadas diretamente.

---

## 4. Tecnologias e ferramentas

| Tecnologia | Finalidade |
|---|---|
| Python | Implementação do pipeline |
| OpenCV | Transformações e filtros |
| NumPy | Manipulação de matrizes e pixels |
| Matplotlib | Visualização comparativa |
| Pandas | Registro das informações e resultados |
| Google Colab | Ambiente de execução |
| Google Drive | Armazenamento dos datasets e imagens processadas |

As versões das bibliotecas utilizadas deverão ser registradas para garantir a reprodutibilidade.

---

## 5. Metodologia de pré-processamento

O pipeline será construído em etapas independentes, permitindo avaliar o efeito de cada transformação.

### 5.1. Leitura das imagens

As imagens serão carregadas a partir das versões finais dos datasets.

Antes do processamento, serão verificados:

- Existência dos arquivos.
- Capacidade de leitura das imagens.
- Dimensões.
- Número de canais.
- Tipo de dados.
- Correspondência com as anotações da Fase 1.

As imagens originais permanecerão disponíveis para comparação.

### 5.2. Conversão para escala de cinza

A primeira transformação será a conversão de imagens coloridas para escala de cinza.

O objetivo é reduzir a representação de três canais de cor para um único canal de intensidade luminosa.

Essa transformação pode facilitar a análise de texturas, bordas e formatos geométricos.

A conversão será realizada utilizando o OpenCV:

```python
imagem_cinza = cv2.cvtColor(
    imagem_original,
    cv2.COLOR_BGR2GRAY
)
```

A conversão para escala de cinza reduz informações de cor. Por isso, sua utilização definitiva dependerá dos resultados dos experimentos posteriores, especialmente se for utilizado um detector pré-treinado que espera imagens RGB.

### 5.3. Redução de ruído

Imagens industriais podem conter pequenas variações de intensidade causadas por captura, compressão ou condições de iluminação.

Para reduzir esses ruídos, será inicialmente avaliado o **filtro de mediana**.

O filtro substitui o valor de cada pixel pela mediana dos valores presentes em sua vizinhança.

A configuração inicial será:

```python
imagem_filtrada = cv2.medianBlur(
    imagem_cinza,
    3
)
```

O parâmetro `3` representa uma vizinhança de 3 × 3 pixels.

A escolha inicial de uma janela pequena busca reduzir ruídos sem suavizar excessivamente contornos e detalhes dos objetos.

Serão observados possíveis efeitos negativos, como perda de pequenos elementos, preenchimento de espaços e alteração de bordas.

### 5.4. Correção de iluminação

A iluminação irregular pode dificultar a distinção entre objetos e fundo.

Serão investigadas técnicas de normalização luminosa para situações em que existam sombras ou diferenças significativas de brilho.

Esses métodos serão aplicados em experimentos separados, para que seus efeitos possam ser comparados ao pipeline básico.

Não será aplicada automaticamente uma correção de iluminação a todas as imagens sem avaliação prévia.

### 5.5. Melhoria de contraste

A melhoria de contraste poderá facilitar a diferenciação entre regiões de intensidade semelhante.

Serão avaliadas duas abordagens:

**Equalização global de histograma**

Redistribui os níveis de intensidade da imagem com o objetivo de ampliar o contraste global.

```python
imagem_equalizada = cv2.equalizeHist(
    imagem_cinza
)
```

**CLAHE — Equalização Adaptativa de Histograma**

Permite ajustar o contraste localmente, utilizando limites para reduzir a amplificação excessiva de ruídos.

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

imagem_clahe = clahe.apply(
    imagem_cinza
)
```

Essas duas abordagens serão tratadas como alternativas experimentais, não como etapas obrigatórias do pipeline final.

---

## 6. Pipeline inicial proposto

A configuração inicial selecionada para os primeiros experimentos será:

**Imagem original → Escala de cinza → Filtro de mediana 3 × 3 → Imagem processada**

Esse pipeline foi escolhido por sua simplicidade, baixo custo computacional e preservação das dimensões espaciais da imagem.

A correção de iluminação e a equalização de histograma serão avaliadas como alternativas complementares.

O pipeline poderá ser modificado conforme os resultados observados.

---

## 7. Estratégia de comparação

As técnicas serão avaliadas individualmente e em combinações controladas.

| Experimento | Transformações | Objetivo |
|---|---|---|
| E0 | Imagem original | Referência de comparação |
| E1 | Escala de cinza | Avaliar o efeito da remoção de cor |
| E2 | Escala de cinza + mediana 3 × 3 | Avaliar redução de ruído |
| E3 | Escala de cinza + equalização global | Avaliar contraste global |
| E4 | Escala de cinza + CLAHE | Avaliar contraste local |

Inicialmente, serão realizadas comparações visuais.

Posteriormente, caso exista um método de detecção e contagem disponível, os experimentos poderão ser comparados utilizando métricas de desempenho sobre o mesmo conjunto de teste.

O objetivo é identificar se as transformações contribuem efetivamente para melhorar o reconhecimento e a contagem, em vez de considerar apenas a aparência visual das imagens.

---

## 8. Preservação das anotações

Uma preocupação essencial da Fase 2 será manter a correspondência entre as imagens processadas e o Ground Truth da Fase 1.

As transformações inicialmente propostas alteram valores de intensidade, mas não modificam as posições dos pixels.

Portanto, quando realizadas sem redimensionamento, recorte ou alteração geométrica, as coordenadas das anotações podem ser preservadas.

Caso sejam introduzidas transformações geométricas futuramente, será necessário atualizar as anotações de forma correspondente.

Os arquivos originais e os rótulos associados deverão permanecer disponíveis para auditoria.

---

## 9. Organização dos resultados

As imagens processadas e os registros experimentais serão armazenados separadamente dos datasets originais.

Estrutura proposta:

```text
PDI_Contagem_Objetos/
│
├── datasets/
│   ├── parafusos/
│   ├── porcas/
│   ├── arruelas/
│   └── vergalhoes/
│
└── resultados/
    ├── fase1_multidataset/
    │
    └── fase2_preprocessamento/
        └── <identificador_da_execucao>/
            ├── comparacoes/
            ├── imagens_processadas/
            ├── metricas.csv
            └── resumo.json
```

A saída de cada execução deverá registrar os parâmetros utilizados, as transformações aplicadas, a quantidade de imagens processadas e eventuais falhas.

---

## 10. Critérios de aceitação

A implementação da Fase 2 será considerada concluída quando:

1. Todas as imagens selecionadas para processamento forem lidas corretamente.
2. As transformações forem executadas sem erros.
3. A quantidade de imagens processadas corresponder à quantidade esperada.
4. As dimensões originais forem preservadas no pipeline básico.
5. A correspondência entre imagens e anotações for mantida.
6. Os datasets originais permanecerem inalterados.
7. Os parâmetros de processamento forem registrados.
8. Forem geradas comparações entre imagens originais e processadas.
9. Os experimentos e suas conclusões forem documentados.
10. O notebook puder ser executado novamente a partir de uma sessão limpa.

A escolha de um pipeline para utilização no treinamento deverá considerar também seu impacto na detecção e contagem, quando essas avaliações estiverem disponíveis.

---

## 11. Limitações e cuidados

Algumas considerações deverão orientar a implementação:

- A escala de cinza pode eliminar informações úteis de cor.
- Filtros de suavização podem remover detalhes pequenos.
- A equalização pode aumentar a visibilidade de ruídos.
- Melhorias visuais não garantem melhorias na detecção.
- A avaliação não deve utilizar imagens de teste para escolher parâmetros.
- Os mesmos procedimentos deverão ser aplicados de forma consistente entre treinamento e avaliação.
- O dataset de vergalhões continuará insuficiente para uma avaliação estatística confiável.
- As limitações de possíveis imagens semelhantes entre conjuntos identificadas na Fase 1 permanecem relevantes.

---

## 12. Estado atual da fase

**Situação: metodologia definida; implementação e validação experimental pendentes.**

O pipeline inicial foi estabelecido, assim como as alternativas de pré-processamento que serão investigadas.

A etapa seguinte consiste em implementar o notebook, executar os experimentos, registrar as evidências e verificar o atendimento aos critérios de aceitação.

Após essa validação, o documento deverá ser atualizado para identificar as técnicas efetivamente utilizadas, seus parâmetros finais e os resultados obtidos.

---

*Projeto acadêmico de Processamento Digital de Imagens — Fase 2: Pré-processamento de Imagens.*
