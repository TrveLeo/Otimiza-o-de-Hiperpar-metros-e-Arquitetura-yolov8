import optuna
import optunahub
from ultralytics import YOLO
import os
import torch
from pathlib import Path



RUNS = "/home/leandrobaldan/Desenvolvimento/Leandro_python/runs/detect"

#arquivos de pesos pré-treinados
MODEL = '/home/leandrobaldan/Desenvolvimento/Leandro_python/compose_train/guarapari_antigo0/weights/best.pt'
MODEL2 = '/home/leandrobaldan/Desenvolvimento/Leandro_python/comparações/best_antigo/weights/best.pt'
MODEL3 = '/home/leandrobaldan/Desenvolvimento/Leandro_python/compose_train/compose_train21/weights/best.pt'
MODEL4 = '/home/leandrobaldan/Desenvolvimento/Leandro_python/comparações/HP_ARCH_OP_NOVO/HP_ARCH_OP_NOVO/weights/last.pt'
MODEL5 = '/home/leandrobaldan/Desenvolvimento/Leandro_python/compose_train/compose_train98/weights/best.pt'
#pasta que contém os dados de treinamento
PASTA = '/home/leandrobaldan/Desenvolvimento/Leandro_python/compose_train'


#arquivos dos datasets
CAMIN = "/home/leandrobaldan/Desenvolvimento/guarapari_test/guarapari.yaml"
CAMIN2 = '/home/leandrobaldan/Desenvolvimento/guarapari_ovo/data.yaml'
CAMIN3 = '/home/leandrobaldan/Desenvolvimento/z/data.yaml'



#função objetiva
def objective(trial):
    model = YOLO(MODEL5)

    #define os parâmetros que seram otimizados 
    lr0 = trial.suggest_float('lr0', 0.000001, 0.1, log=True)
    momentum = trial.suggest_float('momentum', 0.1, 0.99, log=True)
    weightdecay = trial.suggest_float('weight_decay', 0.0001, 0.5, log=True)


    #inicia o treinamento
    train = model.train(data= CAMIN3,
                        epochs=150,
                        device = 0, 
                        project = PASTA,
                        name = f'compose_train',
                        lr0=lr0, 
                        momentum=momentum, 
                        weight_decay=weightdecay, 
                        batch=4)
    

    #inicia a validação
    results = model.val(data=CAMIN3,
                    device=0,
                    plots=True,
                    split="test")
    
    
    #extrai as métricas de avaliação
    metricas = results.results_dict

    precision = metricas['metrics/precision(B)']
    recall = metricas['metrics/recall(B)']
    map50 = metricas['metrics/mAP50(B)']
    map5095 = metricas['metrics/mAP50-95(B)']


    #imprime as métricas
    print('\n'*10)
    print(precision)
    print(recall)
    print(map50)
    print(map5095)
    print('\n'*10)

    #calcula o valor da função
    fitness = ((precision**2 + recall**2 + map50**2 + map5095**2)/4)**0.5
    
    #registra em um arquivo a tentativa
    with open('trials.txt', 'a') as file:
        file.write(f'{lr0}, {momentum}, {weightdecay}, {fitness}\n')
   
    return fitness



study = optuna.create_study(direction="maximize")

#Lê arquivo de trials
with open('trials.txt', 'r') as file:
    for i in file:
        i = i.strip().split(', ')

        #Carrega as trials anteriores
        study.add_trial( 
            optuna.trial.create_trial(
                params={
                    'lr0': float(i[0]),
                    'momentum': float(i[1]),
                    'weight_decay': float(i[2]) 
                },
                distributions={
                    'lr0': optuna.distributions.FloatDistribution(0.000001, 0.1, log=True),
                    'momentum': optuna.distributions.FloatDistribution(0.1, 0.99, log=True),
                    'weight_decay': optuna.distributions.FloatDistribution(0.0001, 0.5, log=True)
                },
                value=float(i[3]) 
            )
        )


study.optimize(objective, n_trials=5)

print("Melhores parâmetros:", study.best_params)