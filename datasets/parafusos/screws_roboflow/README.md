# Dataset de Parafusos — Screws Roboflow

## Descrição

Dataset utilizado no projeto de Processamento Digital
de Imagens para identificação e contagem de parafusos.

## Classes

- DST-A1-35
- DST-A2-70
- N-A2-70
- SCM435
- YFS-109

## Organização

- original/: imagens e anotações originais.
- processado/: imagens resultantes da Fase 2.
- metadata/: Ground Truth e informações do dataset.

## Processamento

Fase 1: Organização e construção do Ground Truth.

Fase 2: Experimentos de pré-processamento e seleção
do pipeline de escala de cinza e filtro de mediana 3x3.

## Observações

As imagens originais devem ser preservadas.

As anotações YOLO devem manter seus identificadores
de classe até a etapa de padronização dos datasets.

A origem, versão e licença do dataset devem ser
verificadas antes da distribuição dos arquivos.
