# Fase 2 — Resultados do Pré-processamento de Imagens

## 1. Introdução

Este documento apresenta os resultados obtidos na **Fase 2 — Pré-processamento de Imagens** do projeto acadêmico de Processamento Digital de Imagens (PDI) — Contagem de Objetos Industriais.

O objetivo desta etapa foi implementar e avaliar visualmente técnicas de pré-processamento, gerando versões tratadas das imagens e garantindo sua sincronização com o Ground Truth validado na Fase 1.

Foram realizados experimentos com conversão para escala de cinza, filtragem por mediana, correção de iluminação, equalização global de histograma e CLAHE.

A avaliação incluiu comparações visuais e verificações automáticas de integridade estrutural.

**Status:** processamento e sincronização estrutural aprovados.

---

## 2. Identificação da execução

| Informação | Valor |
|---|---|
| Etapa | Fase 2 — Pré-processamento |
| Data da execução | 26/11/2026 |
| Ambiente | Google Colab |
| Linguagem | Python |
| Biblioteca principal | OpenCV |
| Datasets processados | 4 |
| Imagens processadas | 975 |
| Instâncias anotadas preservadas | 10.037 |

### Diretório dos resultados

Os arquivos gerados foram armazenados no Google Drive:

`PDI_Contagem_Objetos/resultados/fase2_preprocessamento/20261126T013732688908Z/`

A execução foi realizada com a versão revisada da técnica de correção de iluminação.

---

## 3. Integração com a Fase 1

Antes do processamento, os arquivos de entrada foram comparados com o manifesto da Fase 1.

O procedimento utilizou hashes SHA-256 para confirmar que os arquivos ZIP correspondiam às versões anteriormente aprovadas.

**Manifesto de referência:**

`resultados/fase1_multidataset/20261126T001918539709Z/manifesto_execucao.json`

### Resultado da sincronização

| Dataset | Arquivo de entrada | Verificação SHA-256 |
|---|---|---|
| Roboflow Screws | `revisado.zip` | Aprovada |
| NutsNBolts | `filtrado.zip` | Aprovada |
| Washers Hardee | `deduplicado.zip` | Aprovada |
| Rebar Counting V1 | `original.zip` | Aprovada |

**Resultado:** os quatro arquivos de entrada foram identificados como correspondentes às versões utilizadas e validadas na Fase 1.

Essa verificação permitiu garantir a rastreabilidade dos dados utilizados no pré-processamento.

---

## 4. Resultados do processamento

Todas as imagens selecionadas foram processadas com o pipeline principal:

**Imagem original → Escala de cinza → Filtro de mediana 3 × 3 → Imagem processada**

### 4.1. Quantidade de imagens

| Dataset | Versão de entrada | Imagens processadas |
|---|---|---:|
| Roboflow Screws | `revisado.zip` | 463 |
| NutsNBolts | `filtrado.zip` | 107 |
| Washers Hardee | `deduplicado.zip` | 402 |
| Rebar Counting V1 | `original.zip` | 3 |
| **Total** | | **975** |

O processamento contemplou 100% das imagens previstas nas versões selecionadas.

### 4.2. Preservação do Ground Truth

| Dataset | Objetos anotados na Fase 1 | Objetos preservados na Fase 2 |
|---|---:|---:|
| Roboflow Screws | 3.230 | 3.230 |
| NutsNBolts | 2.832 | 2.832 |
| Washers Hardee | 3.949 | 3.949 |
| Rebar Counting V1 | 26 | 26 |
| **Total** | **10.037** | **10.037** |

A quantidade de instâncias anotadas permaneceu inalterada.

Os arquivos de anotação YOLO também foram preservados, mantendo a correspondência entre imagens, classes e objetos.

---

## 5. Preservação da estrutura dos datasets

O processamento foi projetado para alterar somente os valores dos pixels.

Não foram aplicadas transformações geométricas no pipeline principal.

As seguintes características foram verificadas:

- Quantidade de imagens;
- Quantidade de instâncias anotadas;
- Distribuição entre treinamento, validação e teste;
- Integridade dos arquivos de anotação;
- Dimensões das imagens;
- Existência dos arquivos ZIP de saída.

### Distribuição preservada

| Dataset | Treinamento | Validação | Teste | Total |
|---|---:|---:|---:|---:|
| Roboflow Screws | 387 | 36 | 40 | 463 |
| NutsNBolts | 97 | 5 | 5 | 107 |
| Washers Hardee | 280 | 80 | 42 | 402 |
| Rebar Counting V1 | 3 | 0 | 0 | 3 |
| **Total** | **767** | **121** | **87** | **975** |

As divisões originais foram mantidas.

O dataset de vergalhões continua sem conjuntos de validação e teste, conforme registrado na Fase 1.

---

## 6. Técnicas avaliadas

A implementação contemplou seis representações comparativas.

| Experimento | Técnica | Papel |
|---|---|---|
| E0 | Imagem original | Referência |
| E1 | Escala de cinza | Experimento |
| E2 | Escala de cinza + mediana 3 × 3 | Pipeline principal |
| E3 | Correção de iluminação | Experimento |
| E4 | Equalização global | Experimento |
| E5 | CLAHE | Experimento |

As transformações foram comparadas visualmente em amostras dos datasets.

O objetivo foi observar efeitos sobre intensidade, ruído, contraste e preservação de características visuais dos objetos.

---

## 7. Resultados das comparações visuais

### 7.1. Conversão para escala de cinza

A conversão para escala de cinza preservou os principais contornos e a organização espacial dos objetos nas amostras analisadas.

Em imagens de peças metálicas sobre fundos claros, as diferenças visuais entre a imagem original e a versão em cinza foram relativamente pequenas.

A transformação reduziu a representação da imagem para um único canal de intensidade.

**Observação:** a remoção de informações cromáticas pode ser desfavorável em tarefas nas quais a cor contribui para a identificação dos objetos.

### 7.2. Filtro de mediana 3 × 3

A aplicação do filtro de mediana produziu suavização discreta nas amostras analisadas.

Os contornos principais de parafusos, porcas, arruelas e estruturas metálicas permaneceram visualmente reconhecíveis.

A escolha de uma janela 3 × 3 buscou evitar alterações excessivas.

Entretanto, detalhes pequenos podem ser suavizados, dependendo de seu tamanho e contraste.

### 7.3. Correção de iluminação

A primeira versão do experimento de correção de iluminação produziu escurecimento excessivo em uma amostra do dataset NutsNBolts.

Após a identificação desse comportamento, a implementação foi revisada.

A nova abordagem passou a utilizar uma referência de iluminação derivada da própria imagem, com compensação suave e limitação de ganho.

Na comparação posterior com uma imagem do dataset Rebar Counting V1, o resultado apresentou luminosidade visualmente mais equilibrada que a abordagem anterior.

A correção revisada não foi incorporada ao pipeline principal e permaneceu disponível como alternativa experimental.

**Conclusão visual:** a alteração corrigiu o comportamento de escurecimento excessivo observado anteriormente na amostra de comparação, mas não demonstra melhoria de desempenho em detecção.

### 7.4. Equalização global de histograma

A equalização global aumentou significativamente o contraste em algumas imagens.

Na amostra do NutsNBolts, foram observados realce excessivo de bordas e artefatos, além de regiões com perda aparente de detalhes.

Na amostra de vergalhões, a equalização intensificou tanto as estruturas metálicas quanto as texturas do solo.

**Conclusão visual:** a equalização global não foi considerada adequada como transformação obrigatória para todos os datasets.

### 7.5. CLAHE

O CLAHE apresentou aumento de contraste local mais controlado que a equalização global na amostra de NutsNBolts.

Na imagem de vergalhões, também realçou a textura do solo, além das estruturas metálicas.

Esse comportamento demonstra que o aumento de contraste pode destacar elementos que não pertencem às classes de interesse.

**Conclusão visual:** o CLAHE permanece como alternativa para experimentos específicos, sem aplicação indiscriminada.

---

## 8. Decisão sobre o pipeline principal

O pipeline definido para geração das imagens processadas foi:

**Escala de cinza + filtro de mediana 3 × 3**

A escolha foi baseada em:

- Simplicidade de implementação;
- Baixo custo computacional;
- Ausência de alterações geométricas;
- Suavização moderada;
- Preservação visual dos contornos principais nas amostras;
- Compatibilidade estrutural com as anotações existentes.

A correção de iluminação, a equalização global e o CLAHE permaneceram disponíveis para comparação.

**Importante:** o pipeline foi aprovado como procedimento de pré-processamento e geração de dados, não como solução comprovadamente superior para reconhecimento ou contagem.

A eficiência dessas técnicas deverá ser avaliada quantitativamente utilizando modelos de detecção e conjuntos de teste apropriados.

---

## 9. Critérios de aceitação

Foram executadas verificações automáticas para garantir a consistência dos datasets processados.

| Critério | Arruelas | Parafusos | NutsNBolts | Vergalhões |
|---|---|---|---|---|
| Quantidade de imagens | Aprovado | Aprovado | Aprovado | Aprovado |
| Quantidade de instâncias | Aprovado | Aprovado | Aprovado | Aprovado |
| Distribuição dos conjuntos | Aprovado | Aprovado | Aprovado | Aprovado |
| Anotações idênticas | Aprovado | Aprovado | Aprovado | Aprovado |
| Dimensões preservadas | Aprovado | Aprovado | Aprovado | Aprovado |
| ZIP resultante presente | Aprovado | Aprovado | Aprovado | Aprovado |

Todos os critérios retornaram `True` para os quatro datasets.

### Resultado final

**FASE 2: PROCESSAMENTO E SINCRONIZAÇÃO ESTRUTURAL APROVADOS**

A execução apresentou:

- 975 imagens processadas;
- 10.037 objetos anotados preservados;
- Quatro datasets processados;
- Nenhuma divergência detectada nos critérios de aceitação;
- Arquivos ZIP processados gerados.

---

## 10. Arquivos gerados

Os resultados foram armazenados no Google Drive, no diretório:

`PDI_Contagem_Objetos/resultados/fase2_preprocessamento/20261126T013732688908Z/`

Entre os resultados produzidos estão:

- Versões processadas dos quatro datasets;
- Arquivos ZIP de saída;
- Painéis de comparação visual;
- Relatórios de integridade e sincronização;
- Tabelas consolidadas;
- Informações da execução.

Os datasets aprovados na Fase 1 não foram sobrescritos.

O registro por diretório de execução permite manter o histórico dos experimentos.

---

## 11. Limitações identificadas

Embora todos os critérios estruturais tenham sido aprovados, algumas limitações permanecem.

### 11.1. Avaliação predominantemente visual

A comparação entre técnicas foi realizada por inspeção visual de amostras.

Não foram apresentadas métricas quantitativas de melhoria de reconhecimento.

### 11.2. Ausência de avaliação preditiva

A Fase 2 não contemplou treinamento nem avaliação de modelos de detecção.

Assim, não é possível afirmar que as imagens processadas produzem maior precisão de identificação ou menor erro de contagem.

### 11.3. Perda de informação cromática

A conversão para escala de cinza elimina informações de cor, que podem ser importantes para modelos de reconhecimento.

### 11.4. Suavização de detalhes

O filtro de mediana pode reduzir a visibilidade de elementos pequenos e características relevantes dos objetos.

### 11.5. Realce do fundo

A equalização global e o CLAHE podem aumentar o contraste de texturas e elementos que não pertencem às classes de interesse.

### 11.6. Limitações herdadas da Fase 1

Continuam relevantes:

- A quantidade insuficiente de imagens de vergalhões;
- O tamanho reduzido dos conjuntos de validação e teste do NutsNBolts;
- A necessidade de investigar possíveis imagens semelhantes entre conjuntos;
- A padronização de classes antes do treinamento conjunto.

---

## 12. Recomendações para a próxima fase

Antes de utilizar as imagens processadas em um sistema de reconhecimento e contagem, recomenda-se:

1. Comparar o desempenho de modelos utilizando imagens originais e processadas.
2. Aplicar as mesmas condições de avaliação para ambos os conjuntos.
3. Manter treinamento, validação e teste separados.
4. Definir uma estratégia de padronização das classes entre os datasets.
5. Verificar a compatibilidade entre os formatos YOLO Detection e Segmentation.
6. Avaliar métricas de detecção e erro de contagem.
7. Investigar possíveis diferenças de desempenho entre as técnicas testadas.
8. Ampliar o conjunto de imagens de vergalhões.

As imagens processadas deverão ser tratadas como versões experimentais, preservando o acesso aos dados originais.

---

## 13. Conclusão

A Fase 2 implementou e executou técnicas de pré-processamento sobre os quatro datasets selecionados durante a Fase 1.

O pipeline principal de conversão para escala de cinza seguida de filtro de mediana 3 × 3 foi aplicado às 975 imagens.

As 10.037 instâncias anotadas foram preservadas, assim como os arquivos de rótulos, as dimensões das imagens e os conjuntos de treinamento, validação e teste.

Os experimentos visuais permitiram comparar diferentes técnicas e identificar limitações da equalização global, do CLAHE e da abordagem inicial de correção de iluminação.

Após a revisão da correção de iluminação, a execução final foi novamente validada.

**Todos os critérios automáticos de aceitação foram aprovados.**

Dessa forma, a **Fase 2 está concluída quanto à implementação, ao processamento e à sincronização estrutural**.

A avaliação quantitativa do impacto das transformações sobre o reconhecimento e a contagem de objetos será realizada nas próximas etapas do projeto.

---

**Projeto:** Contagem de Objetos Industriais  
**Disciplina:** Processamento Digital de Imagens  
**Etapa:** Fase 2 — Pré-processamento  
**Execução final:** 26/11/2026  
**Status:** Aprovado estruturalmente
