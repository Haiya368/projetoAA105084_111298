#!/usr/bin/env python3
"""
MAIN.PY - Sistema de Simulação SMA com Labirinto e Farol
Versão completa com treino automático de 10.000 episódios
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Dezembro 2024
"""
import sys
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG_TREINO
from ambiente import AmbienteLabirinto, AmbienteFarol
from agente import AgenteRL, AgenteFixo, GestorAgentes
from sensor import SensorVisao3x3, SensorDirecional

def treinar_agente_labirinto():
    """Função para treinar um agente no labirinto 10x10."""
    
    print("\n" + "="*60)
    print("TREINO DE AGENTE RL NO LABIRINTO 10x10")
    print("="*60)
    
    # Usar configurações do config.py para 10x10
    config = CONFIG_TREINO['labirinto']
    DIMENSAO = (10, 10)  # TABULEIRO 10x10
    EPISODIOS = 200
    MAX_PASSOS = 300
    
    # Criar ambiente
    ambiente = AmbienteLabirinto(DIMENSAO)
    
    # Criar agente com configurações otimizadas
    config_agente = {
        'alfa': 0.1,
        'gama': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    agente = AgenteRL("Agente_RL_Labirinto", (0, 0), **config_agente)
    agente.sensor = SensorVisao3x3()
    
    # Histórico de resultados
    historico = {
        'episodio': [],
        'recompensa': [],
        'passos': [],
        'sucesso': [],
        'epsilon': []
    }
    
    # TREINO
    print(f"\n🧠 Iniciando treino de {EPISODIOS} episódios...")
    print("   (Pressione Ctrl+C para interromper mais cedo)\n")
    
    try:
        for episodio in range(EPISODIOS):
            # Resetar ambiente e agente
            ambiente.posicoes_agentes = {'Agente_RL_Labirinto': (0, 0)}
            ambiente.terminados.clear()
            ambiente.celulas_visitadas.clear()
            agente.reiniciar()
            agente.posicao = (0, 0)
            
            # Executar episódio
            recompensa_episodio = 0
            passos_episodio = 0
            sucesso = False
            
            for passo in range(MAX_PASSOS):
                # Observar ambiente
                obs = ambiente.observacaoPara('Agente_RL_Labirinto')
                if obs is None or obs.get('TERMINOU', False):
                    break
                    
                agente.observacao(obs)
                
                # Escolher ação
                acao = agente.age(modo='aprendizagem')
                
                # Executar ação
                resultado = ambiente.agir(acao, 'Agente_RL_Labirinto')
                
                # Aprender
                prox_obs = ambiente.observacaoPara('Agente_RL_Labirinto')
                agente.avaliacaoEstadoAtual(resultado['recompensa'])
                
                # Atualizar estatísticas
                recompensa_episodio += resultado['recompensa']
                passos_episodio += 1
                
                # Verificar se terminou
                if resultado.get('terminou', False) or resultado.get('episodio_completo', False):
                    sucesso = True
                    break
            
            # Guardar resultados do episódio
            historico['episodio'].append(episodio)
            historico['recompensa'].append(recompensa_episodio)
            historico['passos'].append(passos_episodio)
            historico['sucesso'].append(1 if sucesso else 0)
            historico['epsilon'].append(agente.epsilon)
            
            # Atualizar epsilon
            agente.atualizar_melhor(episodio)
            
            # Progresso
            if (episodio + 1) % 20 == 0:
                ultimos_20 = historico['sucesso'][-20:] if len(historico['sucesso']) >= 20 else historico['sucesso']
                taxa_sucesso = sum(ultimos_20) / len(ultimos_20) * 100 if ultimos_20 else 0
                print(f"  Episódio {episodio+1:3d}/{EPISODIOS} | "
                      f"Recompensa: {recompensa_episodio:7.1f} | "
                      f"Sucesso: {taxa_sucesso:5.1f}% | "
                      f"Epsilon: {agente.epsilon:.3f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo utilizador.")
    
    print(f"\n✅ Treino concluído!")
    print(f"   Total de episódios: {len(historico['episodio'])}")
    taxa_final = sum(historico['sucesso'])/len(historico['sucesso'])*100 if historico['sucesso'] else 0
    print(f"   Taxa de sucesso final: {taxa_final:.1f}%")
    melhor = max(historico['recompensa']) if historico['recompensa'] else 0
    print(f"   Melhor recompensa: {melhor:.1f}")
    
    return historico, agente

def treinar_agente_farol():
    """Função para treinar um agente no problema do Farol."""
    
    print("\n" + "="*60)
    print("TREINO DE AGENTE RL NO FAROL 8x8")
    print("="*60)
    
    # Configurações
    DIMENSAO = (8, 8)
    EPISODIOS = 150
    MAX_PASSOS = 100
    NUM_OBSTACULOS = 8
    
    # Criar ambiente
    ambiente = AmbienteFarol(DIMENSAO, num_obstaculos=NUM_OBSTACULOS)
    
    # Criar agente
    config = {
        'alfa': 0.1,
        'gama': 0.9,
        'epsilon': 1.0,
        'epsilon_decay': 0.98,
        'epsilon_min': 0.05
    }
    
    agente = AgenteRL("Agente_RL_Farol", (0, 0), **config)
    agente.sensor = SensorDirecional()
    
    # Histórico de resultados
    historico = {
        'episodio': [],
        'recompensa': [],
        'passos': [],
        'sucesso': [],
        'epsilon': []
    }
    
    # TREINO
    print(f"\n🧠 Iniciando treino de {EPISODIOS} episódios...")
    print(f"   Obstáculos: {NUM_OBSTACULOS}")
    print("   (Pressione Ctrl+C para interromper mais cedo)\n")
    
    try:
        for episodio in range(EPISODIOS):
            # Resetar ambiente e agente
            ambiente.posicoes_agentes = {'Agente_RL_Farol': (0, 0)}
            ambiente.terminados.clear()
            ambiente.distancias_anteriores.clear()
            agente.reiniciar()
            agente.posicao = (0, 0)
            
            # Executar episódio
            recompensa_episodio = 0
            passos_episodio = 0
            sucesso = False
            
            for passo in range(MAX_PASSOS):
                # Observar ambiente
                obs = ambiente.observacaoPara('Agente_RL_Farol')
                if obs is None or obs.get('TERMINOU', False):
                    break
                    
                agente.observacao(obs)
                
                # Escolher ação
                acao = agente.age(modo='aprendizagem')
                
                # Executar ação
                resultado = ambiente.agir(acao, 'Agente_RL_Farol')
                
                # Aprender
                prox_obs = ambiente.observacaoPara('Agente_RL_Farol')
                agente.avaliacaoEstadoAtual(resultado['recompensa'])
                
                # Atualizar estatísticas
                recompensa_episodio += resultado['recompensa']
                passos_episodio += 1
                
                # Verificar se terminou
                if resultado.get('terminou', False) or resultado.get('episodio_completo', False):
                    sucesso = True
                    break
            
            # Guardar resultados do episódio
            historico['episodio'].append(episodio)
            historico['recompensa'].append(recompensa_episodio)
            historico['passos'].append(passos_episodio)
            historico['sucesso'].append(1 if sucesso else 0)
            historico['epsilon'].append(agente.epsilon)
            
            # Atualizar epsilon
            agente.atualizar_melhor(episodio)
            
            # Progresso
            if (episodio + 1) % 15 == 0:
                ultimos_15 = historico['sucesso'][-15:] if len(historico['sucesso']) >= 15 else historico['sucesso']
                taxa_sucesso = sum(ultimos_15) / len(ultimos_15) * 100 if ultimos_15 else 0
                print(f"  Episódio {episodio+1:3d}/{EPISODIOS} | "
                      f"Recompensa: {recompensa_episodio:7.1f} | "
                      f"Sucesso: {taxa_sucesso:5.1f}% | "
                      f"Epsilon: {agente.epsilon:.3f}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Treino interrompido pelo utilizador.")
    
    print(f"\n✅ Treino concluído!")
    print(f"   Total de episódios: {len(historico['episodio'])}")
    taxa_final = sum(historico['sucesso'])/len(historico['sucesso'])*100 if historico['sucesso'] else 0
    print(f"   Taxa de sucesso final: {taxa_final:.1f}%")
    melhor = max(historico['recompensa']) if historico['recompensa'] else 0
    print(f"   Melhor recompensa: {melhor:.1f}")
    
    return historico, agente

def treinar_todos_agentes_automaticamente():
    """Executa todos os treinos automaticamente com 10.000 episódios."""
    
    print("\n" + "="*60)
    print("TREINO AUTOMÁTICO COMPLETO - 10.000 EPISÓDIOS")
    print("="*60)
    print("⚠️  ATENÇÃO: Este processo pode demorar vários minutos")
    print("    Serão executados:")
    print("    1. Labirinto 10x10: 10.000 episódios")
    print("    2. Farol 8x8: 10.000 episódios")
    print("    Os agentes serão salvos automaticamente")
    print("="*60)
    
    confirmacao = input("\nDeseja continuar? (s/n): ").strip().lower()
    if confirmacao != 's':
        print("Operação cancelada.")
        return None
    
    import time
    tempo_inicio_total = time.time()
    
    # Criar diretórios se não existirem
    os.makedirs('agentes_salvos', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    
    resultados = {}
    
    # 1. TREINO NO LABIRINTO 10x10 (10.000 episódios)
    print("\n" + "="*60)
    print("1. TREINO NO LABIRINTO 10x10 (10.000 episódios)")
    print("="*60)
    
    try:
        tempo_inicio = time.time()
        
        # Configurações para treino rápido
        DIMENSAO = (10, 10)
        EPISODIOS = 20000000
        MAX_PASSOS = 300
        
        # Criar ambiente
        ambiente = AmbienteLabirinto(DIMENSAO)
        
        # Criar agente com configurações otimizadas
        config_agente = {
            'alfa': 0.1,
            'gama': 0.95,
            'epsilon': 1.0,
            'epsilon_decay': 0.9997,  # Mais lento para 10.000 episódios
            'epsilon_min': 0.01
        }
        
        agente = AgenteRL("Agente_RL_Labirinto_10K", (0, 0), **config_agente)
        agente.sensor = SensorVisao3x3()
        
        # Histórico de resultados
        historico = {
            'episodio': [],
            'recompensa': [],
            'passos': [],
            'sucesso': [],
            'epsilon': []
        }
        
        print(f"🧠 Iniciando treino de {EPISODIOS} episódios...")
        print("   (Esta operação pode demorar alguns minutos)")
        
        # TREINO RÁPIDO (com progresso a cada 500 episódios)
        for episodio in range(EPISODIOS):
            # Resetar ambiente e agente
            ambiente.posicoes_agentes = {'Agente_RL_Labirinto_10K': (0, 0)}
            ambiente.terminados.clear()
            ambiente.celulas_visitadas.clear()
            agente.reiniciar()
            agente.posicao = (0, 0)
            
            # Executar episódio
            recompensa_episodio = 0
            passos_episodio = 0
            sucesso = False
            
            for passo in range(MAX_PASSOS):
                # Observar ambiente
                obs = ambiente.observacaoPara('Agente_RL_Labirinto_10K')
                if obs is None or obs.get('TERMINOU', False):
                    break
                    
                agente.observacao(obs)
                
                # Escolher ação
                acao = agente.age(modo='aprendizagem')
                
                # Executar ação
                resultado = ambiente.agir(acao, 'Agente_RL_Labirinto_10K')
                
                # Aprender
                prox_obs = ambiente.observacaoPara('Agente_RL_Labirinto_10K')
                agente.avaliacaoEstadoAtual(resultado['recompensa'])
                
                # Atualizar estatísticas
                recompensa_episodio += resultado['recompensa']
                passos_episodio += 1
                
                # Verificar se terminou
                if resultado.get('terminou', False) or resultado.get('episodio_completo', False):
                    sucesso = True
                    break
            
            # Guardar resultados do episódio
            historico['episodio'].append(episodio)
            historico['recompensa'].append(recompensa_episodio)
            historico['passos'].append(passos_episodio)
            historico['sucesso'].append(1 if sucesso else 0)
            historico['epsilon'].append(agente.epsilon)
            
            # Atualizar epsilon
            agente.atualizar_melhor(episodio)
            
            # Progresso a cada 500 episódios (para não sobrecarregar o console)
            if (episodio + 1) % 500 == 0:
                # Calcular estatísticas dos últimos 500 episódios
                inicio = max(0, episodio - 499)
                sucessos_recentes = historico['sucesso'][inicio:episodio+1]
                taxa_sucesso = sum(sucessos_recentes) / len(sucessos_recentes) * 100
                
                tempo_decorrido = time.time() - tempo_inicio
                tempo_medio_episodio = tempo_decorrido / (episodio + 1)
                tempo_restante = tempo_medio_episodio * (EPISODIOS - episodio - 1)
                
                print(f"  Episódio {episodio+1:5d}/{EPISODIOS} | "
                      f"Sucesso: {taxa_sucesso:5.1f}% | "
                      f"Epsilon: {agente.epsilon:.4f} | "
                      f"Tempo restante: {tempo_restante/60:.1f} min")
        
        # Calcular estatísticas finais
        tempo_final = time.time() - tempo_inicio
        taxa_sucesso_final = sum(historico['sucesso']) / len(historico['sucesso']) * 100
        recompensa_maxima = max(historico['recompensa'])
        
        print(f"\n✅ Treino do Labirinto concluído!")
        print(f"   Tempo total: {tempo_final/60:.1f} minutos")
        print(f"   Taxa de sucesso final: {taxa_sucesso_final:.1f}%")
        print(f"   Melhor recompensa: {recompensa_maxima:.1f}")
        print(f"   Estados aprendidos: {len(agente.q_tabela)}")
        
        # Salvar agente automaticamente
        nome_arquivo = 'agentes_salvos/melhor_labirinto_10K.pkl'
        agente.salvar_agente(nome_arquivo)
        print(f"   ✅ Agente salvo em: {nome_arquivo}")
        
        # Salvar histórico em CSV
        df = pd.DataFrame(historico)
        nome_csv = f'resultados/labirinto_10K_{int(time.time())}.csv'
        df.to_csv(nome_csv, index=False)
        print(f"   📊 Dados salvos em: {nome_csv}")
        
        resultados['labirinto'] = {
            'historico': historico,
            'agente': agente,
            'tempo': tempo_final,
            'arquivo': nome_arquivo
        }
        
    except Exception as e:
        print(f"❌ Erro no treino do Labirinto: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. TREINO NO FAROL 8x8 (10.000 episódios)
    print("\n" + "="*60)
    print("2. TREINO NO FAROL 8x8 (10.000 episódios)")
    print("="*60)
    
    try:
        tempo_inicio = time.time()
        
        # Configurações para treino rápido
        DIMENSAO = (8, 8)
        EPISODIOS = 20000000
        MAX_PASSOS = 100
        NUM_OBSTACULOS = 8
        
        # Criar ambiente
        ambiente = AmbienteFarol(DIMENSAO, num_obstaculos=NUM_OBSTACULOS)
        
        # Criar agente com configurações otimizadas
        config_agente = {
            'alfa': 0.1,
            'gama': 0.9,
            'epsilon': 1.0,
            'epsilon_decay': 0.9997,  # Mais lento para 10.000 episódios
            'epsilon_min': 0.05
        }
        
        agente = AgenteRL("Agente_RL_Farol_10K", (0, 0), **config_agente)
        agente.sensor = SensorDirecional()
        
        # Histórico de resultados
        historico = {
            'episodio': [],
            'recompensa': [],
            'passos': [],
            'sucesso': [],
            'epsilon': []
        }
        
        print(f"🧠 Iniciando treino de {EPISODIOS} episódios...")
        print("   (Esta operação pode demorar alguns minutos)")
        
        # TREINO RÁPIDO (com progresso a cada 500 episódios)
        for episodio in range(EPISODIOS):
            # Resetar ambiente e agente
            ambiente.posicoes_agentes = {'Agente_RL_Farol_10K': (0, 0)}
            ambiente.terminados.clear()
            ambiente.distancias_anteriores.clear()
            agente.reiniciar()
            agente.posicao = (0, 0)
            
            # Executar episódio
            recompensa_episodio = 0
            passos_episodio = 0
            sucesso = False
            
            for passo in range(MAX_PASSOS):
                # Observar ambiente
                obs = ambiente.observacaoPara('Agente_RL_Farol_10K')
                if obs is None or obs.get('TERMINOU', False):
                    break
                    
                agente.observacao(obs)
                
                # Escolher ação
                acao = agente.age(modo='aprendizagem')
                
                # Executar ação
                resultado = ambiente.agir(acao, 'Agente_RL_Farol_10K')
                
                # Aprender
                prox_obs = ambiente.observacaoPara('Agente_RL_Farol_10K')
                agente.avaliacaoEstadoAtual(resultado['recompensa'])
                
                # Atualizar estatísticas
                recompensa_episodio += resultado['recompensa']
                passos_episodio += 1
                
                # Verificar se terminou
                if resultado.get('terminou', False) or resultado.get('episodio_completo', False):
                    sucesso = True
                    break
            
            # Guardar resultados do episódio
            historico['episodio'].append(episodio)
            historico['recompensa'].append(recompensa_episodio)
            historico['passos'].append(passos_episodio)
            historico['sucesso'].append(1 if sucesso else 0)
            historico['epsilon'].append(agente.epsilon)
            
            # Atualizar epsilon
            agente.atualizar_melhor(episodio)
            
            # Progresso a cada 500 episódios
            if (episodio + 1) % 500 == 0:
                # Calcular estatísticas dos últimos 500 episódios
                inicio = max(0, episodio - 499)
                sucessos_recentes = historico['sucesso'][inicio:episodio+1]
                taxa_sucesso = sum(sucessos_recentes) / len(sucessos_recentes) * 100
                
                tempo_decorrido = time.time() - tempo_inicio
                tempo_medio_episodio = tempo_decorrido / (episodio + 1)
                tempo_restante = tempo_medio_episodio * (EPISODIOS - episodio - 1)
                
                print(f"  Episódio {episodio+1:5d}/{EPISODIOS} | "
                      f"Sucesso: {taxa_sucesso:5.1f}% | "
                      f"Epsilon: {agente.epsilon:.4f} | "
                      f"Tempo restante: {tempo_restante/60:.1f} min")
        
        # Calcular estatísticas finais
        tempo_final = time.time() - tempo_inicio
        taxa_sucesso_final = sum(historico['sucesso']) / len(historico['sucesso']) * 100
        recompensa_maxima = max(historico['recompensa'])
        
        print(f"\n✅ Treino do Farol concluído!")
        print(f"   Tempo total: {tempo_final/60:.1f} minutos")
        print(f"   Taxa de sucesso final: {taxa_sucesso_final:.1f}%")
        print(f"   Melhor recompensa: {recompensa_maxima:.1f}")
        print(f"   Estados aprendidos: {len(agente.q_tabela)}")
        
        # Salvar agente automaticamente
        nome_arquivo = 'agentes_salvos/melhor_farol_10K.pkl'
        agente.salvar_agente(nome_arquivo)
        print(f"   ✅ Agente salvo em: {nome_arquivo}")
        
        # Salvar histórico em CSV
        df = pd.DataFrame(historico)
        nome_csv = f'resultados/farol_10K_{int(time.time())}.csv'
        df.to_csv(nome_csv, index=False)
        print(f"   📊 Dados salvos em: {nome_csv}")
        
        resultados['farol'] = {
            'historico': historico,
            'agente': agente,
            'tempo': tempo_final,
            'arquivo': nome_arquivo
        }
        
    except Exception as e:
        print(f"❌ Erro no treino do Farol: {e}")
        import traceback
        traceback.print_exc()
    
    # Resumo final
    tempo_total = time.time() - tempo_inicio_total
    
    print("\n" + "="*60)
    print("✅ TREINO COMPLETO CONCLUÍDO!")
    print("="*60)
    print(f"Tempo total: {tempo_total/60:.1f} minutos")
    
    if 'labirinto' in resultados:
        lab = resultados['labirinto']
        print(f"\n📊 RESULTADOS LABIRINTO 10x10:")
        print(f"   Taxa de sucesso: {sum(lab['historico']['sucesso'])/len(lab['historico']['sucesso'])*100:.1f}%")
        print(f"   Tempo: {lab['tempo']/60:.1f} min")
        print(f"   Agente salvo: {lab['arquivo']}")
    
    if 'farol' in resultados:
        far = resultados['farol']
        print(f"\n📊 RESULTADOS FAROL 8x8:")
        print(f"   Taxa de sucesso: {sum(far['historico']['sucesso'])/len(far['historico']['sucesso'])*100:.1f}%")
        print(f"   Tempo: {far['tempo']/60:.1f} min")
        print(f"   Agente salvo: {far['arquivo']}")
    
    print("\n⚠️  NOTA: Para usar os agentes nas demonstrações,")
    print("   execute o script 'setup_agentes.py' ou")
    print("   copie manualmente os arquivos .pkl:")
    print("   cp agentes_salvos/melhor_labirinto_10K.pkl agentes_salvos/melhor_labirinto.pkl")
    print("   cp agentes_salvos/melhor_farol_10K.pkl agentes_salvos/melhor_farol.pkl")
    
    return resultados

def plotar_resultados(historico, tipo_ambiente: str):
    """Plotar gráficos de resultados."""
    
    # Criar DataFrame
    df = pd.DataFrame(historico)
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Curva de Aprendizagem - Agente RL no {tipo_ambiente.capitalize()}', 
                 fontsize=16, fontweight='bold')
    
    # Cores específicas para cada ambiente
    cores = {
        'labirinto': {'primaria': '#4ECDC4', 'secundaria': '#1A7F78'},
        'farol': {'primaria': '#FF6B6B', 'secundaria': '#CC0000'}
    }
    cor = cores.get(tipo_ambiente, {'primaria': 'blue', 'secundaria': 'red'})
    
    # 1. Recompensa por episódio
    ax1 = axes[0, 0]
    recompensa_suavizada = df['recompensa'].rolling(window=10, min_periods=1).mean()
    
    ax1.plot(df['episodio'], df['recompensa'], alpha=0.3, color=cor['primaria'], label='Original')
    ax1.plot(df['episodio'], recompensa_suavizada, linewidth=2, color=cor['secundaria'], 
             label='Média móvel (10)')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa Total')
    ax1.set_title('Recompensa por Episódio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Taxa de sucesso
    ax2 = axes[0, 1]
    janela = 15 if tipo_ambiente == 'farol' else 20
    taxa_sucesso = df['sucesso'].rolling(window=janela, min_periods=1).mean() * 100
    ax2.plot(df['episodio'], taxa_sucesso, linewidth=2, color='#1DD1A1')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel(f'Taxa de Sucesso (%)\n(média {janela} episódios)')
    ax2.set_title('Taxa de Sucesso')
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    
    # 3. Passos por episódio
    ax3 = axes[1, 0]
    passos_suavizados = df['passos'].rolling(window=10, min_periods=1).mean()
    
    ax3.plot(df['episodio'], df['passos'], alpha=0.3, color='#FF9F43', label='Original')
    ax3.plot(df['episodio'], passos_suavizados, linewidth=2, color='#CC7A00', 
             label='Média móvel (10)')
    ax3.set_xlabel('Episódio')
    ax3.set_ylabel('Número de Passos')
    ax3.set_title('Eficiência (Passos por Episódio)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Evolução do Epsilon
    ax4 = axes[1, 1]
    ax4.plot(df['episodio'], df['epsilon'], linewidth=2, color='#5F27CD')
    ax4.set_xlabel('Episódio')
    ax4.set_ylabel('Valor do Epsilon')
    ax4.set_title('Exploração vs Exploração')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    os.makedirs('resultados', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f'resultados/curva_aprendizagem_{tipo_ambiente}_{timestamp}.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    
    # Salvar dados em CSV
    nome_csv = f'resultados/dados_treinamento_{tipo_ambiente}_{timestamp}.csv'
    df.to_csv(nome_csv, index=False)
    
    print(f"\n📊 Gráfico salvo em: {nome_arquivo}")
    print(f"📄 Dados salvos em: {nome_csv}")
    
    # Mostrar gráfico com tratamento de erro
    try:
        plt.show(block=False)
        print("   Gráfico mostrado. Feche a janela para continuar...")
        plt.pause(2)
    except Exception as e:
        print(f"   Nota: {e}")
        plt.close()
    
    return df

def teste_final(agente_treinado, tipo_ambiente: str):
    """Teste final com o agente treinado."""
    
    print(f"\n" + "="*60)
    print(f"TESTE FINAL - Agente Treinado ({tipo_ambiente.upper()})")
    print("="*60)
    
    # Configurações
    if tipo_ambiente == 'labirinto':
        DIMENSAO = (10, 10)  # TABULEIRO 10x10
        MAX_PASSOS = 100
        ambiente = AmbienteLabirinto(DIMENSAO)
        id_agente = 'Agente_Treinado_Lab'
    else:  # farol
        DIMENSAO = (8, 8)
        MAX_PASSOS = 50
        ambiente = AmbienteFarol(DIMENSAO, num_obstaculos=8)
        id_agente = 'Agente_Treinado_Farol'
    
    ambiente.posicoes_agentes = {id_agente: (0, 0)}
    
    # Configurar agente para modo de teste (epsilon = 0)
    agente_treinado.epsilon = 0.0
    agente_treinado.reiniciar()
    agente_treinado.posicao = (0, 0)
    
    print(f"🧭 Executando teste com política gananciosa...")
    
    # Executar teste
    trajetoria = [(0, 0)]
    recompensa_total = 0
    
    for passo in range(MAX_PASSOS):
        # Observar ambiente
        obs = ambiente.observacaoPara(id_agente)
        if obs is None or obs.get('TERMINOU', False):
            print("  Episódio terminado.")
            break
        
        agente_treinado.observacao(obs)
        
        # Escolher ação (modo teste - sempre ganancioso)
        acao = agente_treinado.age(modo='teste')
        
        # Executar ação
        resultado = ambiente.agir(acao, id_agente)
        recompensa_total += resultado['recompensa']
        
        # Atualizar posição
        nova_posicao = resultado['posicao_nova']
        trajetoria.append(nova_posicao)
        
        # Verificar se chegou ao objetivo
        if resultado.get('terminou', False) or resultado.get('episodio_completo', False):
            objetivo = "saída" if tipo_ambiente == 'labirinto' else "farol"
            print(f"  ✅ Agente encontrou a {objetivo} em {passo+1} passos!")
            print(f"  📍 Últimas posições: {trajetoria[-5:] if len(trajetoria) > 5 else trajetoria}")
            print(f"  💰 Recompensa total: {recompensa_total:.1f}")
            sucesso = True
            break
        
        if passo == MAX_PASSOS - 1:
            print(f"  ⚠️  Teste terminou sem encontrar objetivo após {MAX_PASSOS} passos.")
            print(f"  📍 Últimas posições: {trajetoria[-5:] if len(trajetoria) > 5 else trajetoria}")
            print(f"  💰 Recompensa total: {recompensa_total:.1f}")
            sucesso = False
    
    # Mostrar estatísticas do agente
    stats = agente_treinado.obter_estatisticas_aprendizagem()
    print(f"\n📈 Estatísticas do Agente:")
    print(f"   Estados aprendidos: {stats['estados_aprendidos']}")
    print(f"   Explorações totais: {stats['exploracoes_total']}")
    print(f"   Episódios treinados: {agente_treinado.episodios_treinados}")
    print(f"   Melhor recompensa: {agente_treinado.melhor_recompensa:.1f}")
    print(f"   Epsilon atual: {agente_treinado.epsilon:.3f}")
    
    return sucesso

def comparar_agentes():
    """Compara agente RL com agente fixo em labirinto 10x10."""
    
    print("\n" + "="*60)
    print("COMPARAÇÃO: Agente RL vs Agente Fixo (Labirinto 10x10)")
    print("="*60)
    
    # Configurações
    DIMENSAO = (10, 10)  # TABULEIRO 10x10
    EPISODIOS_COMPARACAO = 30
    MAX_PASSOS = 100
    
    # Criar ambiente Labirinto
    ambiente = AmbienteLabirinto(DIMENSAO)
    
    # Criar agentes
    config_rl = {
        'alfa': 0.1,
        'gama': 0.95,
        'epsilon': 0.1,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01
    }
    
    agente_rl = AgenteRL("Agente_RL", (0, 0), **config_rl)
    agente_rl.sensor = SensorVisao3x3()
    
    agente_fixo = AgenteFixo("Agente_Fixo", (0, 0))
    agente_fixo.sensor = SensorVisao3x3()
    
    # Testar ambos
    resultados = {'RL': [], 'Fixo': []}
    
    for i, (nome, agente) in enumerate([('RL', agente_rl), ('Fixo', agente_fixo)]):
        print(f"\n🧪 Testando {nome}...")
        
        for episodio in range(EPISODIOS_COMPARACAO):
            # Resetar
            ambiente.posicoes_agentes = {f'Agente_{nome}': (0, 0)}
            ambiente.terminados.clear()
            agente.reiniciar()
            agente.posicao = (0, 0)
            
            # Executar episódio
            recompensa_episodio = 0
            sucesso = False
            
            for passo in range(MAX_PASSOS):
                obs = ambiente.observacaoPara(f'Agente_{nome}')
                if obs is None or obs.get('TERMINOU', False):
                    break
                
                agente.observacao(obs)
                acao = agente.age(modo='teste')
                resultado = ambiente.agir(acao, f'Agente_{nome}')
                recompensa_episodio += resultado['recompensa']
                
                if resultado.get('terminou', False):
                    sucesso = True
                    break
            
            resultados[nome].append({
                'recompensa': recompensa_episodio,
                'sucesso': 1 if sucesso else 0,
                'episodio': episodio
            })
        
        # Calcular estatísticas
        recompensas = [r['recompensa'] for r in resultados[nome]]
        sucessos = [r['sucesso'] for r in resultados[nome]]
        
        print(f"  Recompensa média: {np.mean(recompensas):.1f}")
        print(f"  Taxa de sucesso: {np.mean(sucessos)*100:.1f}%")
        print(f"  Melhor recompensa: {np.max(recompensas):.1f}")
    
    # Plotar comparação
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Comparação: Agente RL vs Agente Fixo (Labirinto 10x10)', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Recompensa média
    ax1 = axes[0]
    recompensa_rl = [r['recompensa'] for r in resultados['RL']]
    recompensa_fixo = [r['recompensa'] for r in resultados['Fixo']]
    
    ax1.plot(range(EPISODIOS_COMPARACAO), recompensa_rl, label='Agente RL', color='#4ECDC4', linewidth=2)
    ax1.plot(range(EPISODIOS_COMPARACAO), recompensa_fixo, label='Agente Fixo', color='#FF6B6B', linewidth=2)
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensa por Episódio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Taxa de sucesso acumulada
    ax2 = axes[1]
    sucesso_rl = np.cumsum([r['sucesso'] for r in resultados['RL']])
    sucesso_fixo = np.cumsum([r['sucesso'] for r in resultados['Fixo']])
    
    ax2.plot(range(EPISODIOS_COMPARACAO), sucesso_rl, label='Agente RL', color='#4ECDC4', linewidth=2)
    ax2.plot(range(EPISODIOS_COMPARACAO), sucesso_fixo, label='Agente Fixo', color='#FF6B6B', linewidth=2)
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Sucessos Acumulados')
    ax2.set_title('Sucessos Acumulados')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar figura
    os.makedirs('resultados', exist_ok=True)
    plt.savefig('resultados/comparacao_agentes_10x10.png', dpi=300, bbox_inches='tight')
    
    print(f"\n📊 Gráfico de comparação salvo em: resultados/comparacao_agentes_10x10.png")
    
    try:
        plt.show(block=False)
        print("   Gráfico mostrado. Feche a janela para continuar...")
        plt.pause(2)
    except:
        plt.close()

def carregar_agente_treinado(tipo_ambiente: str):
    """Carrega um agente treinado."""
    
    print(f"\n" + "="*60)
    print(f"CARREGANDO AGENTE TREINADO ({tipo_ambiente.upper()})")
    print("="*60)
    
    # Tentar primeiro os agentes 10K, depois os padrão
    arquivos_tentados = [
        f'agentes_salvos/melhor_{tipo_ambiente}_10K.pkl',
        f'agentes_salvos/melhor_{tipo_ambiente}.pkl'
    ]
    
    nome_arquivo = None
    for arquivo in arquivos_tentados:
        if os.path.exists(arquivo):
            nome_arquivo = arquivo
            break
    
    if nome_arquivo is None:
        print(f"❌ Nenhum agente treinado encontrado para {tipo_ambiente}.")
        print(f"   Execute primeiro o treino (opção 1, 2 ou 8).")
        return None
    
    try:
        with open(nome_arquivo, 'rb') as f:
            dados = pickle.load(f)
        
        # Criar agente
        if tipo_ambiente == 'labirinto':
            agente = AgenteRL("Agente_Treinado", (0, 0))
            agente.sensor = SensorVisao3x3()
        else:  # farol
            agente = AgenteRL("Agente_Treinado", (0, 0))
            agente.sensor = SensorDirecional()
        
        # Carregar dados
        agente.carregar_agente(nome_arquivo)
        
        print(f"✅ Agente carregado com sucesso!")
        print(f"   Arquivo: {nome_arquivo}")
        print(f"   Estados aprendidos: {len(agente.q_tabela)}")
        print(f"   Episódios treinados: {agente.episodios_treinados}")
        print(f"   Melhor recompensa: {agente.melhor_recompensa:.1f}")
        
        return agente
        
    except Exception as e:
        print(f"❌ Erro ao carregar agente: {e}")
        return None

def menu_principal():
    """Menu principal do programa."""
    
    print("\n" + "="*60)
    print("SISTEMA DE SIMULAÇÃO MULTI-AGENTE - SMA")
    print("Labirinto: 10x10 | Farol: 8x8")
    print("ISCTE - Agentes Autónomos")
    print("Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)")
    print("="*60)
    
    while True:
        print("\nMENU PRINCIPAL:")
        print("1. Treinar agente RL no Labirinto (10x10)")
        print("2. Treinar agente RL no Farol (8x8)")
        print("3. Carregar e testar agente treinado (Labirinto 10x10)")
        print("4. Carregar e testar agente treinado (Farol 8x8)")
        print("5. Comparar Agente RL vs Agente Fixo (10x10)")
        print("6. Análise de Resultados (Gráficos)")
        print("7. Demonstração Visual (Ver agentes em ação)")
        print("8. Treino Automático Completo (10.000 episódios cada)")
        print("9. Sair")
        
        try:
            opcao = input("\nEscolha uma opção (1-9): ").strip()
            
            if opcao == '1':
                # Treinar no labirinto 10x10
                historico, agente = treinar_agente_labirinto()
                
                # Perguntar se quer salvar
                salvar = input("\n💾 Deseja salvar o agente treinado? (s/n): ").strip().lower()
                if salvar == 's':
                    os.makedirs('agentes_salvos', exist_ok=True)
                    agente.salvar_agente('agentes_salvos/melhor_labirinto.pkl')
                    print("✅ Agente salvo em 'agentes_salvos/melhor_labirinto.pkl'")
                
                # Plotar resultados
                plotar = input("\n📊 Deseja ver os gráficos de aprendizagem? (s/n): ").strip().lower()
                if plotar == 's':
                    df = plotar_resultados(historico, 'labirinto')
                    print("\nPressione Enter para continuar...")
                    input()
                
            elif opcao == '2':
                # Treinar no farol 8x8
                historico, agente = treinar_agente_farol()
                
                # Perguntar se quer salvar
                salvar = input("\n💾 Deseja salvar o agente treinado? (s/n): ").strip().lower()
                if salvar == 's':
                    os.makedirs('agentes_salvos', exist_ok=True)
                    agente.salvar_agente('agentes_salvos/melhor_farol.pkl')
                    print("✅ Agente salvo em 'agentes_salvos/melhor_farol.pkl'")
                
                # Plotar resultados
                plotar = input("\n📊 Deseja ver os gráficos de aprendizagem? (s/n): ").strip().lower()
                if plotar == 's':
                    df = plotar_resultados(historico, 'farol')
                    print("\nPressione Enter para continuar...")
                    input()
                
            elif opcao == '3':
                # Carregar e testar labirinto 10x10
                agente = carregar_agente_treinado('labirinto')
                if agente:
                    teste = input("\n🧪 Deseja testar o agente? (s/n): ").strip().lower()
                    if teste == 's':
                        teste_final(agente, 'labirinto')
                        print("\nPressione Enter para continuar...")
                        input()
                
            elif opcao == '4':
                # Carregar e testar farol 8x8
                agente = carregar_agente_treinado('farol')
                if agente:
                    teste = input("\n🧪 Deseja testar o agente? (s/n): ").strip().lower()
                    if teste == 's':
                        teste_final(agente, 'farol')
                        print("\nPressione Enter para continuar...")
                        input()
                
            elif opcao == '5':
                comparar_agentes()
                print("\nPressione Enter para continuar...")
                input()
                
            elif opcao == '6':
                # Análise de resultados
                try:
                    from analise_resultados import menu_analise_resultados
                    menu_analise_resultados()
                except ImportError as e:
                    print(f"❌ Erro ao importar módulo de análise: {e}")
                    print("   Certifique-se que o arquivo 'analise_resultados.py' está no diretório.")
                    input("\nPressione Enter para continuar...")
                
            elif opcao == '7':
                # Demonstrações visuais
                try:
                    from visualizacao_tempo_real import menu_demonstracoes
                    menu_demonstracoes()
                except ImportError as e:
                    print(f"❌ Erro ao importar módulo de visualização: {e}")
                    print("   Certifique-se que o arquivo 'visualizacao_tempo_real.py' está no diretório.")
                    input("\nPressione Enter para continuar...")
                
            elif opcao == '8':
                # Treino automático completo
                resultados = treinar_todos_agentes_automaticamente()
                
                # Perguntar se quer criar gráficos
                if resultados:
                    plotar = input("\n📊 Deseja gerar gráficos dos resultados? (s/n): ").strip().lower()
                    if plotar == 's':
                        for tipo, dados in resultados.items():
                            if 'historico' in dados:
                                print(f"\nGerando gráficos para {tipo}...")
                                df = pd.DataFrame(dados['historico'])
                                plotar_resultados(dados['historico'], tipo)
                
                print("\nPressione Enter para continuar...")
                input()
                
            elif opcao == '9':
                print("\n👋 Encerrando programa...")
                print("Obrigado por usar o Simulador SMA!")
                break
            
            else:
                print("❌ Opção inválida. Tente novamente.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Operação interrompida pelo utilizador.")
            continuar = input("Deseja sair do programa? (s/n): ").strip().lower()
            if continuar == 's':
                print("👋 Encerrando programa...")
                break
        
        except Exception as e:
            print(f"\n❌ Erro: {e}")
            import traceback
            traceback.print_exc()
            input("\nPressione Enter para continuar...")


if __name__ == "__main__":
    # Criar diretórios necessários
    os.makedirs('agentes_salvos', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    
    menu_principal()