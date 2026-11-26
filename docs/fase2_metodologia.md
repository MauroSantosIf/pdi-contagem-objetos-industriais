# Fase 2 — Metodologia de Pré-processamento de Imagens

## 1. Introdução

A segunda fase do projeto acadêmico **Contagem de Objetos Industriais**, desenvolvido na disciplina de Processamento Digital de Imagens (PDI), teve como objetivo implementar e validar técnicas de pré-processamento de imagens para apoiar as etapas posteriores de reconhecimento e contagem de objetos industriais.

Após a conclusão da Fase 1, responsável pela organização dos datasets, validação estrutural das anotações e construção do Ground Truth, foram aplicadas técnicas de processamento digital de imagens para investigar a redução de ruídos, a correção de iluminação e o aprimoramento de contraste.

A implementação foi desenvolvida em Python, utilizando a biblioteca OpenCV, com execução no Google Colab e armazenamento dos resultados no Google Drive.

O processo foi estruturado para garantir a correspondência entre os datasets originais, as imagens processadas e as anotações aprovadas na Fase 1.

**Situação:** implementação e validação estrutural concluídas.

---

## 2. Objetivos

### 2.1. Objetivo geral

Implementar e avaliar visualmente técnicas de pré-processamento em imagens industriais, preservando sua integridade estrutural e a correspondência com o Ground Truth da Fase 1.

### 2.2. Objetivos específicos

- Processar os quatro datasets selecionados na Fase 1.
- Aplicar conversão para escala de cinza.
- Aplicar filtragem espacial para redução de ruídos.
- Avaliar métodos de correção de iluminação.
- Comparar técnicas de equalização de histograma.
- Gerar comparações entre imagens originais e processadas.
- Preservar as dimensões das imagens.
- Manter os arquivos de anotação YOLO inalterados.
- Verificar a distribuição entre treinamento, validação e teste.
- Gerar datasets processados e relatórios de execução.
- Garantir a rastreabilidade dos arquivos utilizados.

---

## 3. Ambiente de desenvolvimento

As seguintes tecnologias foram utilizadas:

| Tecnologia | Aplicação |
|---|---|
| Python | Implementação das rotinas de pré-processamento |
| OpenCV | Conversão de cores, filtros e transformações |
| NumPy | Manipulação das matrizes de pixels |
| Matplotlib | Visualização e comparação das imagens |
| Pandas | Organização dos resultados e relatórios |
| Google Colab | Execução do notebook |
| Google Drive | Armazenamento dos datasets e resultados |
| SHA-256 | Verificação da integridade dos arquivos de entrada |
| ZIP | Empacotamento dos datasets processados |

O ambiente de execução foi configurado para acessar os datasets já validados durante a Fase 1.

---

## 4. Datasets utilizados

Foram utilizadas exclusivamente as versões aprovadas na Fase 1.

| Dataset | Versão de entrada | Imagens | Objetos anotados |
|---|---|---:|---:|
| Roboflow Screws | `revisado.zip` | 463 | 3.230 |
| NutsNBolts | `filtrado.zip` | 107 | 2.832 |
| Washers Hardee | `deduplicado.zip` | 402 | 3.949 |
| Rebar Counting V1 | `original.zip` | 3 | 26 |
| **Total** | | **975** | **10.037** |

O dataset de vergalhões permaneceu classificado como exploratório, pois possui somente três imagens de treinamento, sem conjuntos de validação e teste.

Os arquivos originais e as versões aprovadas foram preservados durante toda a execução.

---

## 5. Integração com a Fase 1

Uma das principais preocupações da Fase 2 foi garantir que as imagens processadas correspondessem exatamente às imagens e anotações validadas na etapa anterior.

Para isso, o notebook realizou uma verificação dos arquivos ZIP utilizados como entrada.

### 5.1. Verificação por SHA-256

Foi utilizado o algoritmo SHA-256 para calcular a assinatura digital de cada arquivo ZIP.

As assinaturas obtidas foram comparadas com os valores registrados no manifesto de execução da Fase 1.

O manifesto utilizado foi:

`resultados/fase1_multidataset/20261126T001918539709Z/manifesto_execucao.json`

A verificação confirmou que os quatro arquivos de entrada correspondiam às versões aprovadas anteriormente.

**Resultado: sincronização SHA-256 aprovada.**

### 5.2. Preservação do Ground Truth

O pré-processamento foi realizado exclusivamente sobre os valores dos pixels das imagens.

O pipeline principal não introduziu modificações geométricas, como recortes, rotações ou redimensionamentos.

Dessa forma, as dimensões das imagens foram preservadas e as coordenadas normalizadas das anotações YOLO permaneceram compatíveis.

Os arquivos de anotação foram mantidos sem alterações.

Também foram preservados os conjuntos de origem:

- Treinamento (`train`);
- Validação (`valid`);
- Teste (`test`).

---

## 6. Metodologia de pré-processamento

A implementação contemplou cinco técnicas de processamento digital de imagens, avaliadas individualmente ou em combinações específicas.

O objetivo foi comparar diferentes estratégias e selecionar uma configuração inicial conservadora.

### 6.1. Conversão para escala de cinza

A primeira técnica consistiu na conversão das imagens coloridas para escala de cinza.

A transformação representa a informação visual utilizando um único canal de intensidade luminosa.

No OpenCV, a conversão é realizada por:

```python
imagem_cinza = cv2.cvtColor(
    imagem_original,
    cv2.COLOR_BGR2GRAY
)
```

A conversão reduz a quantidade de canais e permite trabalhar diretamente com informações relacionadas a intensidade, textura e contornos.

Entretanto, também elimina as informações cromáticas, que podem ser relevantes para determinadas tarefas de reconhecimento.

Por esse motivo, a imagem original foi preservada para futuras comparações.

### 6.2. Redução de ruído com filtro de mediana

Após a conversão para escala de cinza, foi aplicado o filtro de mediana com uma janela de 3 × 3 pixels.

```python
imagem_filtrada = cv2.medianBlur(
    imagem_cinza,
    3
)
```

O filtro de mediana é uma técnica não linear que substitui o valor de cada pixel pela mediana dos valores presentes em sua vizinhança.

A finalidade é reduzir pequenas variações de intensidade e determinados tipos de ruído, evitando uma suavização excessiva.

A janela 3 × 3 foi escolhida como configuração inicial por produzir uma intervenção relativamente pequena nas imagens.

A aplicação pode suavizar detalhes muito pequenos; portanto, seus efeitos devem ser considerados nas etapas de reconhecimento.

### 6.3. Correção de iluminação

Também foi implementada uma técnica experimental de correção de iluminação.

O objetivo foi compensar diferenças de luminosidade e reduzir efeitos de iluminação não uniforme nas imagens.

A implementação utilizou uma estimativa suave da iluminação de fundo e ajustes controlados de intensidade.

#### Ajuste realizado durante os experimentos

A primeira implementação apresentou escurecimento excessivo em uma amostra do dataset NutsNBolts.

A técnica foi revisada para:

- Utilizar uma referência de iluminação derivada da própria imagem;
- Aplicar uma compensação suave;
- Limitar o ganho da correção;
- Controlar alterações excessivas da luminosidade média.

Após a modificação, uma nova execução foi realizada para verificar os resultados visuais e estruturais.

Na imagem analisada do dataset Rebar Counting V1, a correção revisada apresentou luminosidade mais equilibrada em relação à implementação anterior, preservando os principais contornos visíveis.

A correção de iluminação permaneceu como **alternativa experimental**, não sendo incorporada automaticamente ao pipeline principal de exportação dos datasets.

### 6.4. Equalização global de histograma

A equalização global de histograma foi avaliada como alternativa para redistribuição dos níveis de intensidade.

No OpenCV, essa técnica pode ser aplicada por:

```python
imagem_equalizada = cv2.equalizeHist(
    imagem_cinza
)
```

O objetivo é ampliar o contraste global da imagem.

Entretanto, os experimentos visuais mostraram que essa transformação pode produzir resultados excessivamente contrastados.

Na imagem de NutsNBolts analisada, foram observados:

- Aumento acentuado do contraste;
- Realce de artefatos;
- Regiões excessivamente claras ou escuras;
- Possível perda de detalhes relevantes.

Por esse motivo, a equalização global não foi selecionada como parte obrigatória do pipeline principal.

### 6.5. CLAHE — Equalização Adaptativa de Histograma

A técnica CLAHE (*Contrast Limited Adaptive Histogram Equalization*) também foi avaliada.

Diferentemente da equalização global, o CLAHE atua sobre regiões locais da imagem e limita a amplificação do contraste.

Uma configuração experimental utilizada como referência foi:

```python
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

imagem_clahe = clahe.apply(
    imagem_cinza
)
```

Na comparação visual, o CLAHE apresentou aumento do contraste local, tornando algumas texturas e contornos mais evidentes.

Entretanto, em imagens com fundos muito texturizados, como a amostra de vergalhões, também houve realce significativo de detalhes do solo.

Esse comportamento poderá aumentar a complexidade visual para determinados algoritmos de detecção.

Assim, o CLAHE foi mantido como técnica experimental para estudos posteriores.

---

## 7. Pipeline principal aplicado

Após a organização dos experimentos, foi mantido como pipeline principal:

**Imagem original → Escala de cinza → Filtro de mediana 3 × 3 → Imagem processada**

Essa configuração foi adotada para gerar uma versão padronizada das imagens, com alterações moderadas de intensidade e suavização.

O pipeline não altera a geometria das imagens.

As técnicas de correção de iluminação, equalização global e CLAHE foram utilizadas na etapa comparativa, sem substituir o pipeline principal.

A seleção desse pipeline representa uma decisão metodológica inicial, e não uma comprovação de melhora na precisão de reconhecimento.

---

## 8. Experimentos comparativos

O notebook gerou painéis de comparação contendo seis representações de imagens selecionadas.

| Experimento | Técnica |
|---|---|
| E0 | Imagem original |
| E1 | Escala de cinza |
| E2 | Escala de cinza + mediana 3 × 3 |
| E3 | Correção de iluminação |
| E4 | Equalização global |
| E5 | CLAHE |

Os painéis permitiram observar diferenças visuais entre as transformações.

### 8.1. Amostra NutsNBolts

Na imagem analisada, contendo parafusos, porcas e arruelas sobre fundo predominantemente claro:

- A escala de cinza preservou a estrutura geral dos objetos.
- O filtro de mediana produziu suavização discreta.
- A versão inicial da correção de iluminação apresentou escurecimento excessivo.
- A equalização global gerou contraste muito acentuado.
- O CLAHE apresentou contraste local mais controlado.

Essas observações motivaram a revisão da técnica de correção de iluminação.

### 8.2. Amostra Rebar Counting V1

Na imagem analisada, contendo estruturas metálicas e solo:

- A conversão para cinza preservou os contornos principais.
- O filtro de mediana suavizou variações locais de intensidade.
- A correção de iluminação revisada apresentou luminosidade visualmente equilibrada.
- A equalização global destacou intensamente as texturas do fundo.
- O CLAHE aumentou o contraste local, inclusive em regiões do solo.

A análise indicou que técnicas de aumento de contraste podem realçar tanto os objetos de interesse quanto elementos indesejados do cenário.

### 8.3. Limitação da avaliação visual

As observações foram realizadas sobre amostras selecionadas.

Não foi demonstrado que qualquer uma das técnicas melhore a precisão de um modelo de detecção ou a qualidade das contagens.

Essa conclusão dependerá de experimentos quantitativos nas etapas posteriores do projeto.

---

## 9. Processamento dos datasets

Após a definição do pipeline principal, as imagens dos quatro datasets foram processadas.

Para cada imagem:

1. Foi realizada a leitura do arquivo original.
2. A imagem foi convertida para escala de cinza.
3. Foi aplicado o filtro de mediana 3 × 3.
4. Foi gerada a imagem processada.
5. Foram preservadas as dimensões originais.
6. O arquivo de anotação correspondente foi mantido inalterado.
7. A organização dos conjuntos foi preservada.

Os resultados foram organizados em novos arquivos ZIP, sem sobrescrever os dados da Fase 1.

### 9.1. Quantidade de imagens processadas

| Dataset | Imagens processadas |
|---|---:|
| Roboflow Screws | 463 |
| NutsNBolts | 107 |
| Washers Hardee | 402 |
| Rebar Counting V1 | 3 |
| **Total** | **975** |

### 9.2. Preservação das anotações

| Dataset | Instâncias preservadas |
|---|---:|
| Roboflow Screws | 3.230 |
| NutsNBolts | 2.832 |
| Washers Hardee | 3.949 |
| Rebar Counting V1 | 26 |
| **Total** | **10.037** |

Nenhuma instância anotada foi perdida durante o processamento.

---

## 10. Critérios de aceitação

Foram definidos critérios automáticos para verificar a integridade dos resultados.

| Critério | Finalidade |
|---|---|
| `contagem_imagens_ok` | Conferir a quantidade de imagens |
| `contagem_instancias_ok` | Conferir a quantidade de objetos anotados |
| `splits_ok` | Verificar a preservação dos conjuntos |
| `labels_identicos` | Confirmar que os rótulos não foram modificados |
| `dimensoes_preservadas` | Confirmar a manutenção das dimensões |
| `zip_resultante_presente` | Verificar a geração dos arquivos processados |

### Resultado da execução final

| Dataset | Imagens | Instâncias | Critérios aprovados |
|---|---:|---:|---|
| Arruelas | 402 | 3.949 | 6 de 6 |
| Parafusos | 463 | 3.230 | 6 de 6 |
| NutsNBolts | 107 | 2.832 | 6 de 6 |
| Vergalhões | 3 | 26 | 6 de 6 |

Todos os critérios retornaram `True` para os quatro datasets.

**Resultado: processamento e sincronização estrutural aprovados.**

Esses critérios demonstram que a quantidade de imagens, as instâncias anotadas, os conjuntos e os arquivos de saída foram preservados conforme o esperado.

Eles não constituem métricas de desempenho de reconhecimento.

---

## 11. Armazenamento e rastreabilidade

Os resultados da execução final foram armazenados no Google Drive.

Diretório da execução:

`PDI_Contagem_Objetos/resultados/fase2_preprocessamento/20261126T013732688908Z/`

Esse diretório contém os arquivos gerados e os relatórios associados à execução.

A organização por identificador de execução permite manter registros independentes de diferentes experimentos.

A utilização dos arquivos ZIP aprovados na Fase 1 foi verificada por SHA-256, garantindo que o processamento partisse das versões esperadas.

Os datasets de entrada permaneceram preservados.

---

## 12. Limitações

Apesar da aprovação nos critérios estruturais, a Fase 2 apresenta limitações importantes:

1. A avaliação das técnicas foi predominantemente visual.
2. Não foram medidas melhorias de precisão ou erro de contagem.
3. A escala de cinza elimina informações de cor.
4. O filtro de mediana pode suavizar características pequenas.
5. A equalização global pode aumentar excessivamente o contraste.
6. O CLAHE pode intensificar texturas do fundo.
7. A correção de iluminação apresentou resultados dependentes das características da imagem.
8. O dataset de vergalhões continua insuficiente para uma avaliação quantitativa confiável.
9. A utilização de imagens processadas em modelos pré-treinados pode exigir adaptações ao formato de entrada.

Essas limitações deverão ser consideradas durante a implementação dos algoritmos de reconhecimento e contagem.

---

## 13. Reprodutibilidade

O código da Fase 2 deverá ser mantido no repositório, no arquivo:

`notebooks/fase2_preprocessamento_multidataset_sincronizado.ipynb`

Para reproduzir a execução:

1. Disponibilizar as versões finais dos datasets.
2. Executar o notebook da Fase 1, se necessário, para gerar o manifesto de referência.
3. Abrir o notebook da Fase 2 no Google Colab.
4. Autorizar o acesso à pasta do Google Drive.
5. Confirmar o diretório configurado para os datasets.
6. Executar as células na ordem apresentada.
7. Conferir a sincronização SHA-256.
8. Analisar os painéis comparativos.
9. Conferir os relatórios e os critérios de aceitação.

O notebook utiliza verificações automáticas para detectar divergências estruturais entre as imagens processadas e os dados aprovados anteriormente.

---

## 14. Conclusão

A Fase 2 implementou e avaliou técnicas de pré-processamento de imagens industriais, utilizando os quatro datasets validados na Fase 1.

Foram avaliadas a conversão para escala de cinza, a filtragem por mediana, a correção de iluminação, a equalização global de histograma e o CLAHE.

O pipeline principal selecionado para geração dos datasets processados foi composto por conversão para escala de cinza seguida de filtro de mediana 3 × 3.

A correção de iluminação foi revisada após a identificação de escurecimento excessivo em uma amostra, permanecendo como alternativa experimental.

Ao final, foram processadas **975 imagens**, preservando **10.037 instâncias anotadas**, os conjuntos de treinamento, validação e teste e as dimensões originais das imagens.

Todos os critérios automáticos de aceitação estrutural foram aprovados.

**A Fase 2 está concluída quanto à implementação, processamento e sincronização estrutural.**

A avaliação da influência dessas transformações sobre a precisão de reconhecimento e contagem será realizada nas próximas etapas.

---

**Projeto:** Contagem de Objetos Industriais  
**Disciplina:** Processamento Digital de Imagens  
**Etapa:** Fase 2 — Pré-processamento de Imagens  
**Status:** Implementação e validação estrutural concluídas
