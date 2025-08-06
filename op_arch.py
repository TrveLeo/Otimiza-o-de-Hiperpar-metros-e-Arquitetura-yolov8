import optuna
import optunahub
from ultralytics import YOLO
import os
import torch
import random as rd
import math as mt


# Constantes
CAMIN = "/home/leandrobaldan/Desenvolvimento/guarapari_test/guarapari.yaml"
CAMIN2 = '/home/leandrobaldan/Desenvolvimento/guarapari_ovo/data.yaml'
MODELO = '/home/leandrobaldan/Desenvolvimento/Leandro_python/cods/dados.yaml'
MODELO2 = '/home/leandrobaldan/Desenvolvimento/Leandro_python/cods/runs/detect/train17/weights/best.pt'

T_F = [True, False]
ADD_REMOVE = ['add', 'rem']
EXTRACT = ['Conv', 'C2f', 'SPPF']



# Função de otimização
def objective(trial):
    global ADD_REMOVE, EXTRACT, T_F

    # Carrega o modelo
    model = YOLO('yolov8n.pt')
    yaml = model.model.yaml['backbone']

    # Arquitetura
    num = trial.suggest_int('num', 1, 6)
    n_camadas_back = trial.suggest_int('n_camadas_back', 6, len(model.model.yaml['backbone']))
    n_fil = trial.suggest_int('n_fil', 16, 256)
    len_kernel = trial.suggest_int('len_kernel', 3, 18)
    stride = trial.suggest_int('stride', 1, 3)
    padding = trial.suggest_int('padding', 1, 2)
    extr = trial.suggest_int('extr', 0, 2)
    add_remove_mod = trial.suggest_int('add_remove', 0, 2)
    t_f = trial.suggest_int('t_f', 0, 1)

    #Treinamento
    lr0 = trial.suggest_float('lr0', 1e-6, 0.1, log=True)
    momentum = trial.suggest_float('momentum', 0.1, 0.99, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.5, log=True)

    # Configuração da rede
    last_fil = yaml[-1][3][0] #Guarda o valor do ultimo filtro

    if add_remove_mod == 0:  # Adiciona camada

        if extr == 0: #Verifica qual rede será adcionada e configura de acordo com seus argumentos
            lista = [-1, num, EXTRACT[extr], [last_fil * 2, 3, 2]]
        elif extr == 1:
            lista = [-1, 1, EXTRACT[extr], [last_fil * 2, True]]
        else:
            lista = [-1, num, EXTRACT[extr], [last_fil * 2, 3, 2]]
        yaml.append(lista)

    elif add_remove_mod == 1:  #Modifica camada
        filtro = yaml[n_camadas_back][3][0]
        j = yaml[n_camadas_back][2]
        for i in EXTRACT:
            if i == j:
                jota = i

        if jota == 0:
            lista = [-1, num, EXTRACT[extr], [filtro, len_kernel, stride]]
        elif jota == 1:
            lista = [-1, 1, EXTRACT[extr], [filtro, T_F[t_f]]]
        else:
            lista = [-1, num, EXTRACT[extr], [filtro, 5]]
        yaml[n_camadas_back] = lista



    else:  # Remove camada
        model.model.yaml['backbone'].pop(-1)


    # Treinamento do modelo
    model.train(data=CAMIN,
                epochs=10,
                device=0,
                lr0=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
                batch=8)


    # Avaliação do modelo
    results = model.val(data=CAMIN, device=0, plots=True, split="test")
    metricas = results.results_dict

    # Cálculo de métricas
    precision = metricas['metrics/precision(B)']
    recall = metricas['metrics/recall(B)']
    map50 = metricas['metrics/mAP50(B)']
    map5095 = metricas['metrics/mAP50-95(B)']


    
    with open('tesete.yaml', 'w') as file:
        dados = model.model.yaml

        #Escreve o numero de classes
        file.write(f"nc: {dados['nc']}\n") 
        file.write(f'\n')
        
        #Escreve o tamanho da rede    
        file.write('depth_multiple: 0.33\n')
        file.write('width_multiple: 0.50\n')
        file.write('max_channels: 512\n')
        file.write('\n')
        
            #Escreve a escreve a backbone
        file.write(f'backbone: \n')
        for i in range(len(dados['backbone'])):
            file.write('  - [')
            for j in range(len(dados['backbone'][i])):
                file.write(f"{dados['backbone'][i][j]}, ")
            file.write(']\n')

        file.write(f'\n')
        
        
            #Escreve a head
        file.write(f'head: \n')
        for i in range(len(dados['head'])):
            file.write('  - [')
            for j in range(len(dados['head'][i])):
                file.write(f"'{dados['head'][i][j]}'" if dados['head'][i][j] is str() else f"{dados['head'][i][j]}, ")
            file.write(']\n')

    


    # Retorna a função objetivo
    fitness = ((precision ** 2 + recall ** 2 + map50 ** 2 + map5095 ** 2) / 4) ** 0.5
    return fitness

# Configuração do estudo
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Adiciona trials anteriores
with open('trials.txt', 'r') as file:
    for line in file:
        params = list(map(float, line.strip().split(', ')))
        study.add_trial(
            optuna.trial.create_trial(
                params={
                    'lr0': params[0],
                    'momentum': params[1],
                    'weight_decay': params[2],
                },
                distributions={
                    'lr0': optuna.distributions.FloatDistribution(1e-6, 0.1, log=True),
                    'momentum': optuna.distributions.FloatDistribution(0.1, 0.99, log=True),
                    'weight_decay': optuna.distributions.FloatDistribution(1e-4, 0.5, log=True),
                },
                value=params[3]
            )
        )

# Resultados
print("Melhores parâmetros:", study.best_params)
