#!/usr/bin/env python3
"""
VISUALIZAÇÃO EM TEMPO REAL - Para ver o "boneco a andar"
Versão completa com suporte a agentes treinados com 10.000 episódios
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import os

class VisualizadorTempoReal:
    """Visualizador para ver os agentes em movimento em tabuleiro 10x10."""
    
    def __init__(self, ambiente, titulo="Simulador SMA", velocidade=0.3):
        self.ambiente = ambiente
        self.titulo = titulo
        self.velocidade = velocidade
        self.fig = None
        self.ax = None
        
    def iniciar(self):
        """Inicia a visualização."""
        # Ajustar tamanho da figura baseado no ambiente
        linhas, colunas = self.ambiente.dimensao
        
        # Para ambiente 10x10, usar tamanho maior
        if linhas == 10 and colunas == 10:
            self.fig, self.ax = plt.subplots(figsize=(12, 12))
        else:
            # Para outros tamanhos, calcular proporcionalmente
            tamanho_base = 10
            largura = max(8, tamanho_base * (colunas / 10))
            altura = max(8, tamanho_base * (linhas / 10))
            self.fig, self.ax = plt.subplots(figsize=(largura, altura))
            
        self.fig.canvas.manager.set_window_title(self.titulo)
        
        # Configurar eixos para mostrar todo o quadro
        self.ax.set_xlim(-1, colunas)  # Margem extra nas bordas
        self.ax.set_ylim(-1, linhas)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Para que (0,0) fique no canto superior esquerdo
        
        # Configurar grade
        self.ax.set_xticks(range(colunas))
        self.ax.set_yticks(range(linhas))
        self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Remover labels dos eixos para mais espaço
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Adicionar bordas ao redor do grid
        self.ax.add_patch(patches.Rectangle(
            (-0.5, -0.5), colunas, linhas,
            linewidth=2, edgecolor='black', facecolor='none'
        ))
        
        plt.ion()  # Modo interativo
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Ajustar margens
        plt.show()
        
        print(f"[Visualizador] Figura configurada para {linhas}x{colunas}")
        
    def atualizar(self, passo=None, recompensa_acumulada=0):
        """Atualiza a visualização."""
        if self.ax is None:
            return
            
        self.ax.clear()
        
        linhas, colunas = self.ambiente.dimensao
        
        # Reconfigurar limites dos eixos
        self.ax.set_xlim(-1, colunas)
        self.ax.set_ylim(-1, linhas)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.set_xticks(range(colunas))
        self.ax.set_yticks(range(linhas))
        self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        
        # Desenhar borda do tabuleiro
        self.ax.add_patch(patches.Rectangle(
            (-0.5, -0.5), colunas, linhas,
            linewidth=2, edgecolor='black', facecolor='none'
        ))
        
        # Desenhar fundo branco em todas as células
        for linha in range(linhas):
            for coluna in range(colunas):
                self.ax.add_patch(patches.Rectangle(
                    (coluna - 0.5, linha - 0.5), 1, 1,
                    linewidth=0.5, edgecolor='lightgray', facecolor='white', alpha=0.3
                ))
        
        # Desenhar elementos específicos
        if hasattr(self.ambiente, 'tipo'):
            if self.ambiente.tipo == 'farol':
                self._desenhar_farol()
            elif self.ambiente.tipo == 'labirinto':
                self._desenhar_labirinto()
        
        # Desenhar agentes
        self._desenhar_agentes()
        
        # Título com informações
        titulo_completo = self.titulo
        if passo is not None:
            titulo_completo += f" | Passo: {passo}"
        if recompensa_acumulada != 0:
            titulo_completo += f" | Recompensa: {recompensa_acumulada:.1f}"
        
        # Título menor para não ocupar muito espaço
        self.ax.set_title(titulo_completo, fontsize=12, fontweight='bold', pad=10)
        
        # Atualizar o canvas
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        if self.velocidade > 0:
            time.sleep(self.velocidade)
    
    def _desenhar_farol(self):
        """Desenha elementos do Farol."""
        # Farol - círculo maior e mais visível
        linha_f, coluna_f = self.ambiente.farol
        self.ax.add_patch(patches.Circle(
            (coluna_f, linha_f), 0.35,
            facecolor='yellow', edgecolor='orange', linewidth=3, alpha=0.9
        ))
        self.ax.text(coluna_f, linha_f, 'F', ha='center', va='center', 
                    fontsize=14, fontweight='bold', color='darkorange')
        
        # Obstáculos - mais visíveis
        for obstaculo in self.ambiente.obstaculos:
            linha_o, coluna_o = obstaculo
            self.ax.add_patch(patches.Rectangle(
                (coluna_o - 0.4, linha_o - 0.4), 0.8, 0.8,
                facecolor='black', edgecolor='darkgray', linewidth=2, alpha=0.8
            ))
            self.ax.text(coluna_o, linha_o, 'X', ha='center', va='center',
                        color='white', fontsize=12, fontweight='bold')
    
    def _desenhar_labirinto(self):
        """Desenha elementos do Labirinto."""
        linhas, colunas = self.ambiente.dimensao
        
        # Paredes - mais visíveis
        for linha in range(linhas):
            for coluna in range(colunas):
                if self.ambiente.mapa[linha, coluna] == 1:
                    self.ax.add_patch(patches.Rectangle(
                        (coluna - 0.5, linha - 0.5), 1, 1,
                        facecolor='#606060', edgecolor='#404040', linewidth=2, alpha=0.9
                    ))
                    # Adicionar textura às paredes
                    if (linha + coluna) % 2 == 0:
                        self.ax.text(coluna, linha, '#', ha='center', va='center',
                                    color='#303030', fontsize=10, alpha=0.7)
        
        # Saída (9,9 em tabuleiro 10x10) - mais visível
        linha_s, coluna_s = self.ambiente.saida
        self.ax.add_patch(patches.Rectangle(
            (coluna_s - 0.45, linha_s - 0.45), 0.9, 0.9,
            facecolor='#00CC00', edgecolor='#006600', linewidth=3, alpha=0.9
        ))
        self.ax.text(coluna_s, linha_s, 'S', ha='center', va='center',
                    color='white', fontsize=14, fontweight='bold')
        
        # Entrada (0,0) - mais visível
        self.ax.add_patch(patches.Rectangle(
            (-0.45, -0.45), 0.9, 0.9,
            facecolor='#6495ED', edgecolor='#1E90FF', linewidth=3, alpha=0.9
        ))
        self.ax.text(0, 0, 'E', ha='center', va='center',
                    color='white', fontsize=12, fontweight='bold')
    
    def _desenhar_agentes(self):
        """Desenha todos os agentes."""
        cores = ['#FF4444', '#4444FF', '#8844FF', '#FF8844', '#44FF88', '#FF44FF']
        
        for i, (agente_id, posicao) in enumerate(self.ambiente.posicoes_agentes.items()):
            if posicao is None:
                continue
                
            linha, coluna = posicao
            cor = cores[i % len(cores)]
            
            # Agente (círculo grande e colorido)
            self.ax.add_patch(patches.Circle(
                (coluna, linha), 0.4,
                facecolor=cor, edgecolor='black', linewidth=2, alpha=0.9
            ))
            
            # Identificador - mais visível
            if agente_id:
                # Primeira letra do ID
                inicial = agente_id[0].upper()
                self.ax.text(coluna, linha, inicial, 
                            ha='center', va='center',
                            color='white', fontsize=11, fontweight='bold')
                
                # ID completo ao lado (se houver espaço)
                if len(agente_id) > 1 and self.ambiente.dimensao[1] >= 10:
                    self.ax.text(coluna + 0.5, linha, agente_id[1:], 
                                ha='left', va='center',
                                color='black', fontsize=8, fontweight='normal')
    
    def pausar(self, segundos: float = 2.0):
        """Pausa a visualização por um tempo específico."""
        plt.pause(segundos)
    
    def fechar(self):
        """Fecha a visualização."""
        if self.fig:
            plt.close(self.fig)
            plt.ioff()

def demonstrar_labirinto_agente_fixo():
    """Demonstração do Labirinto 10x10 com agente de regras fixas."""
    from ambiente import AmbienteLabirinto
    from agente import AgenteFixo
    from sensor import SensorVisao3x3
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Labirinto 10x10 - Agente Fixo")
    print("(Algoritmo da mão direita)")
    print("="*60)
    
    # Criar ambiente 10x10
    ambiente = AmbienteLabirinto((10, 10))
    
    # Criar agente fixo
    agente = AgenteFixo("Agente_Fixo", (0, 0))
    agente.sensor = SensorVisao3x3()
    
    # Visualizador com velocidade mais lenta para visualização
    vis = VisualizadorTempoReal(ambiente, "Labirinto 10x10 - Agente Fixo", velocidade=0.4)
    vis.iniciar()
    
    # Posicionar agente
    ambiente.posicoes_agentes = {'Agente_Fixo': (0, 0)}
    
    print("🧭 Agente Fixo iniciando no labirinto 10x10...")
    print("   Estratégia: Seguir parede direita (algoritmo da mão direita)")
    print(f"   Saída: {ambiente.saida}")
    print("   Objetivo: Encontrar a saída (S) partindo da entrada (E)")
    print("\n   Pressione Ctrl+C para interromper...")
    
    # Executar demonstração
    max_passos = 200
    recompensa_total = 0
    
    try:
        for passo in range(max_passos):
            # Observar
            obs = ambiente.observacaoPara('Agente_Fixo')
            if obs is None or obs.get('TERMINOU', False):
                print("  Episódio terminado.")
                break
            
            agente.observacao(obs)
            
            # Agir
            acao = agente.age(modo='teste')
            
            # Executar
            resultado = ambiente.agir(acao, 'Agente_Fixo')
            recompensa_total += resultado['recompensa']
            
            # Atualizar visualização
            vis.atualizar(passo=passo+1, recompensa_acumulada=recompensa_total)
            
            # Mostrar posição atual periodicamente
            if (passo + 1) % 10 == 0:
                pos = ambiente.posicoes_agentes['Agente_Fixo']
                print(f"  Passo {passo+1}: Posição {pos}")
            
            # Verificar sucesso
            if resultado.get('terminou', False):
                print(f"\n   AGENTE ENCONTROU A SAÍDA!")
                print(f"   Conclusão em {passo+1} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                print(f"   Trajetória finalizada com sucesso")
                vis.pausar(3)  # Pausa para ver a posição final
                break
            
            if passo == max_passos - 1:
                print(f"\n    LIMITE DE PASSOS ATINGIDO")
                print(f"   Não encontrou saída em {max_passos} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                pos_final = ambiente.posicoes_agentes['Agente_Fixo']
                distancia = abs(pos_final[0] - ambiente.saida[0]) + abs(pos_final[1] - ambiente.saida[1])
                print(f"   Posição final: {pos_final}")
                print(f"   Distância à saída: {distancia}")
                vis.pausar(3)
    
    except KeyboardInterrupt:
        print("\n\n  Demonstração interrompida pelo utilizador")
    
    print("\nPressione Enter para continuar...")
    input()
    vis.fechar()

def demonstrar_farol_agente_fixo():
    """Demonstração do Farol 8x8 com agente de regras fixas."""
    from ambiente import AmbienteFarol
    from agente import AgenteFixo
    from sensor import SensorDirecional
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Farol 8x8 - Agente Fixo")
    print("(Seguir direção do farol)")
    print("="*60)
    
    # Criar ambiente com menos obstáculos para facilitar
    ambiente = AmbienteFarol((8, 8), num_obstaculos=5)
    
    # Criar agente fixo
    agente = AgenteFixo("Agente_Fixo", (0, 0))
    agente.sensor = SensorDirecional()
    
    # Visualizador
    vis = VisualizadorTempoReal(ambiente, "Farol 8x8 - Agente Fixo", velocidade=0.3)
    vis.iniciar()
    
    # Posicionar agente
    ambiente.posicoes_agentes = {'Agente_Fixo': (0, 0)}
    
    print("   Agente Fixo iniciando no problema do Farol...")
    print("   Estratégia: Seguir direção do farol")
    print(f"   Farol na posição: {ambiente.farol}")
    print(f"   Obstáculos: {len(ambiente.obstaculos)}")
    print("   Objetivo: Chegar ao farol (F) evitando obstáculos (X)")
    print("\n   Pressione Ctrl+C para interromper...")
    
    # Executar demonstração
    max_passos = 80
    recompensa_total = 0
    
    try:
        for passo in range(max_passos):
            # Observar
            obs = ambiente.observacaoPara('Agente_Fixo')
            if obs is None or obs.get('TERMINOU', False):
                print("  Episódio terminado.")
                break
            
            agente.observacao(obs)
            
            # Agir
            acao = agente.age(modo='teste')
            
            # Executar
            resultado = ambiente.agir(acao, 'Agente_Fixo')
            recompensa_total += resultado['recompensa']
            
            # Atualizar visualização
            vis.atualizar(passo=passo+1, recompensa_acumulada=recompensa_total)
            
            # Mostrar direção periodicamente
            if (passo + 1) % 8 == 0 and 'direcao_farol' in obs:
                print(f"  Passo {passo+1}: Direção do farol = {obs['direcao_farol']}")
            
            # Verificar sucesso
            if resultado.get('terminou', False):
                print(f"\n   AGENTE ALCANÇOU O FAROL!")
                print(f"   Conclusão em {passo+1} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                vis.pausar(3)
                break
            
            if passo == max_passos - 1:
                pos_final = ambiente.posicoes_agentes['Agente_Fixo']
                distancia = abs(pos_final[0] - ambiente.farol[0]) + abs(pos_final[1] - ambiente.farol[1])
                print(f"\n    LIMITE DE PASSOS ATINGIDO")
                print(f"   Não alcançou farol em {max_passos} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                print(f"   Posição final: {pos_final}")
                print(f"   Distância ao farol: {distancia}")
                vis.pausar(3)
    
    except KeyboardInterrupt:
        print("\n\n  Demonstração interrompida pelo utilizador")
    
    print("\nPressione Enter para continuar...")
    input()
    vis.fechar()

def demonstrar_labirinto_agente_treinado_arquivo(arquivo_agente):
    """Demonstração do Labirinto 10x10 com agente RL treinado (arquivo específico)."""
    import pickle
    from ambiente import AmbienteLabirinto
    from agente import AgenteRL
    from sensor import SensorVisao3x3
    
    print(f"\n   Carregando agente de: {arquivo_agente}")
    
    try:
        # Tentar carregar usando o método da classe
        agente = AgenteRL("Agente_RL", (0, 0))
        agente.sensor = SensorVisao3x3()
        agente.carregar_agente(arquivo_agente)
        
        # Verificar se carregou corretamente
        if len(agente.q_tabela) == 0:
            print("  Agente carregado mas sem aprendizagem (Q-table vazia)")
    except:
        print(" Erro ao carregar agente. Tentando método alternativo...")
        try:
            with open(arquivo_agente, 'rb') as f:
                agente = pickle.load(f)
        except Exception as e:
            print(f" Erro crítico ao carregar agente: {e}")
            return
    
    # Criar ambiente 10x10
    ambiente = AmbienteLabirinto((10, 10))
    
    # Visualizador com velocidade mais rápida (já que é treinado)
    vis = VisualizadorTempoReal(ambiente, "Labirinto 10x10 - Agente RL Treinado", velocidade=0.2)
    vis.iniciar()
    
    # Configurar agente para teste
    if hasattr(agente, 'epsilon'):
        agente.epsilon = 0.0  # Política gananciosa
    if hasattr(agente, 'reiniciar'):
        agente.reiniciar()
    agente.posicao = (0, 0)
    
    # Posicionar agente
    ambiente.posicoes_agentes = {'Agente_RL': (0, 0)}
    
    print(" Agente RL Treinado iniciando no labirinto 10x10...")
    print("   Modo: Política gananciosa (epsilon=0)")
    print(f"   Estados aprendidos: {len(agente.q_tabela) if hasattr(agente, 'q_tabela') else 'N/A'}")
    print(f"   Episódios treinados: {agente.episodios_treinados if hasattr(agente, 'episódios_treinados') else 'N/A'}")
    print(f"   Saída: {ambiente.saida}")
    print("\n   Pressione Ctrl+C para interromper...")
    
    # Executar demonstração
    max_passos = 100
    recompensa_total = 0
    trajetoria = [(0, 0)]
    
    try:
        for passo in range(max_passos):
            # Observar
            obs = ambiente.observacaoPara('Agente_RL')
            if obs is None or obs.get('TERMINOU', False):
                print("  Episódio terminado.")
                break
            
            agente.observacao(obs)
            
            # Agir (modo teste)
            acao = agente.age(modo='teste')
            
            # Executar
            resultado = ambiente.agir(acao, 'Agente_RL')
            recompensa_total += resultado['recompensa']
            
            # Registrar trajetória
            nova_pos = resultado['posicao_nova']
            if nova_pos not in trajetoria[-1:]:  # Evitar repetir mesma posição
                trajetoria.append(nova_pos)
            
            # Atualizar visualização
            vis.atualizar(passo=passo+1, recompensa_acumulada=recompensa_total)
            
            # Mostrar aprendizado periodicamente
            if (passo + 1) % 20 == 0:
                print(f"  Passo {passo+1}: Posição {nova_pos}")
            
            # Verificar sucesso
            if resultado.get('terminou', False):
                print(f"\n   AGENTE ENCONTROU A SAÍDA!")
                print(f"   Conclusão em {passo+1} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                print(f"   Caminho eficiente aprendido!")
                if len(trajetoria) <= 20:  # Mostrar trajetória se não for muito longa
                    print(f" Trajetória: {trajetoria}")
                vis.pausar(3)
                break
            
            if passo == max_passos - 1:
                print(f"\n    LIMITE DE PASSOS ATINGIDO")
                print(f"   Não encontrou saída em {max_passos} passos")
                print(f"  Recompensa total: {recompensa_total:.1f}")
                print(f"   Última posição: {ambiente.posicoes_agentes['Agente_RL']}")
                if len(trajetoria) > 1:
                    print(f"  🗺️  Últimos 5 passos: {trajetoria[-5:]}")
                vis.pausar(3)
    
    except KeyboardInterrupt:
        print("\n\n  Demonstração interrompida pelo utilizador")
    
    print("\nPressione Enter para continuar...")
    input()
    vis.fechar()

def demonstrar_labirinto_agente_treinado():
    """Demonstração do Labirinto 10x10 com agente RL treinado."""
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Labirinto 10x10 - Agente RL Treinado")
    print("(Política gananciosa aprendida)")
    print("="*60)
    
    # Tentar primeiro carregar o agente 10K, depois o padrão
    arquivos_tentados = [
        'agentes_salvos/melhor_labirinto_10K.pkl',
        'agentes_salvos/melhor_labirinto.pkl'
    ]
    
    arquivo_agente = None
    for arquivo in arquivos_tentados:
        if os.path.exists(arquivo):
            arquivo_agente = arquivo
            break
    
    if arquivo_agente is None:
        print(" Nenhum agente treinado encontrado.")
        print("   Execute primeiro o treino (opção 1 ou 8 no menu principal).")
        return
    
    demonstrar_labirinto_agente_treinado_arquivo(arquivo_agente)

def demonstrar_farol_agente_treinado_arquivo(arquivo_agente):
    """Demonstração do Farol 8x8 com agente RL treinado (arquivo específico)."""
    import pickle
    from ambiente import AmbienteFarol
    from agente import AgenteRL
    from sensor import SensorDirecional
    
    print(f"\n   Carregando agente de: {arquivo_agente}")
    
    try:
        # Tentar carregar usando o método da classe
        agente = AgenteRL("Agente_RL", (0, 0))
        agente.sensor = SensorDirecional()
        agente.carregar_agente(arquivo_agente)
        
        # Verificar se carregou corretamente
        if len(agente.q_tabela) == 0:
            print("  Agente carregado mas sem aprendizagem (Q-table vazia)")
    except:
        print(" Erro ao carregar agente. Tentando método alternativo...")
        try:
            with open(arquivo_agente, 'rb') as f:
                agente = pickle.load(f)
        except Exception as e:
            print(f" Erro crítico ao carregar agente: {e}")
            return
    
    # Criar ambiente
    ambiente = AmbienteFarol((8, 8), num_obstaculos=6)
    
    # Visualizador
    vis = VisualizadorTempoReal(ambiente, "Farol 8x8 - Agente RL Treinado", velocidade=0.2)
    vis.iniciar()
    
    # Configurar agente para teste
    if hasattr(agente, 'epsilon'):
        agente.epsilon = 0.0  # Política gananciosa
    if hasattr(agente, 'reiniciar'):
        agente.reiniciar()
    agente.posicao = (0, 0)
    
    # Posicionar agente
    ambiente.posicoes_agentes = {'Agente_RL': (0, 0)}
    
    print(" Agente RL Treinado iniciando no problema do Farol...")
    print("   Modo: Política gananciosa (epsilon=0)")
    print(f"   Estados aprendidos: {len(agente.q_tabela) if hasattr(agente, 'q_tabela') else 'N/A'}")
    print(f"   Episódios treinados: {agente.episodios_treinados if hasattr(agente, 'episodios_treinados') else 'N/A'}")
    print(f"   Farol na posição: {ambiente.farol}")
    print(f"   Obstáculos: {len(ambiente.obstaculos)}")
    print("\n   Pressione Ctrl+C para interromper...")
    
    # Executar demonstração
    max_passos = 50
    recompensa_total = 0
    distancias = []
    
    try:
        for passo in range(max_passos):
            # Observar
            obs = ambiente.observacaoPara('Agente_RL')
            if obs is None or obs.get('TERMINOU', False):
                print("  Episódio terminado.")
                break
            
            agente.observacao(obs)
            
            # Agir (modo teste)
            acao = agente.age(modo='teste')
            
            # Executar
            resultado = ambiente.agir(acao, 'Agente_RL')
            recompensa_total += resultado['recompensa']
            
            # Calcular distância atual ao farol
            pos_atual = ambiente.posicoes_agentes['Agente_RL']
            distancia = abs(pos_atual[0] - ambiente.farol[0]) + abs(pos_atual[1] - ambiente.farol[1])
            distancias.append(distancia)
            
            # Atualizar visualização
            vis.atualizar(passo=passo+1, recompensa_acumulada=recompensa_total)
            
            # Mostrar progresso
            if (passo + 1) % 8 == 0:
                print(f"  Passo {passo+1}: Distância ao farol = {distancia}")
            
            # Verificar sucesso
            if resultado.get('terminou', False):
                print(f"\n   AGENTE ALCANÇOU O FAROL!")
                print(f"   Conclusão em {passo+1} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                print(f"   Melhoria na distância: {distancias[0]} → 0")
                vis.pausar(3)
                break
            
            if passo == max_passos - 1:
                print(f"\n    LIMITE DE PASSOS ATINGIDO")
                print(f"   Não alcançou farol em {max_passos} passos")
                print(f"   Recompensa total: {recompensa_total:.1f}")
                print(f"   Posição final: {pos_atual}")
                print(f"   Distância final ao farol: {distancia}")
                if len(distancias) > 1:
                    print(f"   Melhoria: {distancias[0]} → {distancia}")
                vis.pausar(3)
    
    except KeyboardInterrupt:
        print("\n\n  Demonstração interrompida pelo utilizador")
    
    print("\nPressione Enter para continuar...")
    input()
    vis.fechar()

def demonstrar_farol_agente_treinado():
    """Demonstração do Farol 8x8 com agente RL treinado."""
    
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO: Farol 8x8 - Agente RL Treinado")
    print("(Política gananciosa aprendida)")
    print("="*60)
    
    # Tentar primeiro carregar o agente 10K, depois o padrão
    arquivos_tentados = [
        'agentes_salvos/melhor_farol_10K.pkl',
        'agentes_salvos/melhor_farol.pkl'
    ]
    
    arquivo_agente = None
    for arquivo in arquivos_tentados:
        if os.path.exists(arquivo):
            arquivo_agente = arquivo
            break
    
    if arquivo_agente is None:
        print(" Nenhun agente treinado encontrado.")
        print("   Execute primeiro o treino (opção 2 ou 8 no menu principal).")
        return
    
    demonstrar_farol_agente_treinado_arquivo(arquivo_agente)

def menu_demonstracoes():
    """Menu de demonstrações visuais."""
    
    while True:
        print("\n" + "="*60)
        print("DEMONSTRAÇÕES VISUAIS - Ver agentes em ação")
        print("="*60)
        print("1. Labirinto 10x10 - Agente Fixo")
        print("   (Regras: seguir parede direita)")
        print("2. Farol 8x8 - Agente Fixo")
        print("   (Regras: seguir direção do farol)")
        print("3. Labirinto 10x10 - Agente RL Treinado")
        print("   (Política gananciosa aprendida)")
        print("4. Farol 8x8 - Agente RL Treinado")
        print("   (Política gananciosa aprendida)")
        print("5. Voltar ao menu principal")
        
        try:
            opcao = input("\nEscolha uma demonstração (1-5): ").strip()
            
            if opcao == '1':
                demonstrar_labirinto_agente_fixo()
            elif opcao == '2':
                demonstrar_farol_agente_fixo()
            elif opcao == '3':
                demonstrar_labirinto_agente_treinado()
            elif opcao == '4':
                demonstrar_farol_agente_treinado()
            elif opcao == '5':
                print("\nVoltando ao menu principal...")
                break
            else:
                print("\n Opção inválida. Escolha 1-5.")
                
        except KeyboardInterrupt:
            print("\n\n  Demonstração interrompida pelo utilizador.")
            break
        except Exception as e:
            print(f"\n Erro durante a demonstração: {e}")
            import traceback
            traceback.print_exc()
            print("\nPressione Enter para continuar...")
            input()

if __name__ == "__main__":
    # Criar diretórios necessários
    os.makedirs('agentes_salvos', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)
    
    menu_demonstracoes()