#!/usr/bin/env python3
"""
Configurações Globais do Sistema.
Define hiperparâmetros otimizados para o relatório.
"""

# Configurações de Cores e Estilo
CORES = {
    'reset': '\033[0m',
    'vermelho': '\033[91m',
    'verde': '\033[92m',
    'azul': '\033[94m',
    'amarelo': '\033[93m',
    'negrito': '\033[1m'
}


# Hiperparâmetros de Treino (Ajustados para 10x10)
CONFIG_TREINO = {
    'labirinto': {
        'dimensao': (10, 10),        # TABULEIRO 10x10
        'max_passos': 300,
        'num_episodios': 500,
        'alfa': 0.1,
        'gama': 0.95,
        'epsilon_inicial': 1.0,
        'epsilon_decay': 0.99,
        'epsilon_min': 0.01
    },
    'farol': {
        'dimensao': (8, 8),
        'num_obstaculos': 10,
        'max_passos': 100,
        'num_episodios': 300,
        'alfa': 0.1,
        'gama': 0.9,
        'epsilon_inicial': 1.0,
        'epsilon_decay': 0.98,
        'epsilon_min': 0.01
    }
}