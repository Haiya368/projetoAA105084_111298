
"""
Módulo de visualização para o Simulador SMA.
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Dezembro 2024
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from typing import Tuple, Dict, Any


class Visualizador:
    """Classe para visualização gráfica dos ambientes."""
    
    def __init__(self, ambiente, titulo: str = "Simulador SMA", 
                 velocidade: float = 0.1, mostrar_grade: bool = True,
                 tamanho_celula: float = 40):
        """
        Inicializa o visualizador.
        
        Args:
            ambiente: O ambiente a ser visualizado
            titulo: Título da visualização
            velocidade: Tempo de pausa entre atualizações (segundos)
            mostrar_grade: Se True, mostra a grade do ambiente
            tamanho_celula: Tamanho de cada célula em pixels (afeta o tamanho da figura)
        """
        self.ambiente = ambiente
        self.titulo = titulo
        self.velocidade = velocidade
        self.mostrar_grade = mostrar_grade
        self.tamanho_celula = tamanho_celula
        
        # Calcular dimensões da figura baseado no ambiente
        linhas, colunas = ambiente.dimensao
        largura_figura = max(8, colunas * (tamanho_celula / 100))
        altura_figura = max(8, linhas * (tamanho_celula / 100))
        
        # Inicializar figura do matplotlib com tamanho dinâmico
        self.fig, self.ax = plt.subplots(figsize=(largura_figura, altura_figura))
        self.fig.canvas.manager.set_window_title(titulo)
        
        # Configurar eixos com margens
        self.ax.set_xlim(-0.5, colunas - 0.5)
        self.ax.set_ylim(-0.5, linhas - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Para que (0,0) fique no canto superior esquerdo
        
        # Configurar grade com espaçamento apropriado
        if mostrar_grade:
            self.ax.set_xticks(range(colunas))
            self.ax.set_yticks(range(linhas))
            self.ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        else:
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.grid(False)
        
        # Configurar título e rótulos com tamanho ajustável
        tamanho_titulo = max(10, min(14, 14 - (max(linhas, colunas) - 10) * 0.5))
        self.ax.set_title(titulo, fontsize=tamanho_titulo, pad=15)
        self.ax.set_xlabel("Colunas", fontsize=10)
        self.ax.set_ylabel("Linhas", fontsize=10)
        
        # Ajustar tamanho das fontes dos ticks
        tamanho_ticks = max(8, min(10, 10 - (max(linhas, colunas) - 10) * 0.3))
        self.ax.tick_params(axis='both', which='major', labelsize=tamanho_ticks)
        
        # Elementos gráficos
        self.patches = []
        self.textos = []
        
        # Informações de status
        self.status_text = None
        self.passo_atual = 0
        
        # Criar visualização inicial
        self.atualizar()
        
        # Mostrar a figura (não bloqueante)
        plt.ion()
        plt.tight_layout()
        plt.show()
        
        print(f"[Visualizador] Criada visualização para ambiente {ambiente.tipo}")
        print(f"  Dimensão: {ambiente.dimensao}")
        print(f"  Tamanho da figura: {largura_figura:.1f} x {altura_figura:.1f} polegadas")
        print(f"  Velocidade: {velocidade}s por passo")
        print(f"  Grade: {'Sim' if mostrar_grade else 'Não'}")
    
    def _limpar_figura(self):
        """Remove todos os patches e textos da figura."""
        for patch in self.patches:
            patch.remove()
        for texto in self.textos:
            texto.remove()
        
        self.patches.clear()
        self.textos.clear()
        
        # Remover texto de status se existir
        if self.status_text:
            self.status_text.remove()
            self.status_text = None
    
    def _adicionar_celula(self, linha: int, coluna: int, cor: str, 
                         borda: str = 'black', alpha: float = 1.0,
                         linha_largura: float = 1.0):
        """Adiciona uma célula colorida à visualização."""
        rect = patches.Rectangle(
            (coluna - 0.5, linha - 0.5), 1, 1,
            linewidth=linha_largura, edgecolor=borda, facecolor=cor, alpha=alpha
        )
        self.ax.add_patch(rect)
        self.patches.append(rect)
    
    def _adicionar_texto(self, linha: int, coluna: int, texto: str, 
                        cor: str = 'black', tamanho: int = 10,
                        deslocamento_x: float = 0, deslocamento_y: float = 0):
        """Adiciona texto à visualização."""
        texto_obj = self.ax.text(
            coluna + deslocamento_x, linha + deslocamento_y, texto,
            ha='center', va='center',
            color=cor, fontsize=tamanho, fontweight='bold'
        )
        self.textos.append(texto_obj)
    
    def atualizar(self, passo: int = None, info_agentes: Dict = None):
        """
        Atualiza a visualização com o estado atual do ambiente.
        
        Args:
            passo: Número do passo atual da simulação
            info_agentes: Informações adicionais sobre os agentes
        """
        self._limpar_figura()
        
        if passo is not None:
            self.passo_atual = passo
        
        linhas, colunas = self.ambiente.dimensao
        
        # Desenhar fundo
        for linha in range(linhas):
            for coluna in range(colunas):
                self._adicionar_celula(linha, coluna, 'white', 'lightgray', 0.3)
        
        # Desenhar elementos específicos do ambiente
        if self.ambiente.tipo == 'farol':
            self._desenhar_farol()
        elif self.ambiente.tipo == 'labirinto':
            self._desenhar_labirinto()
        
        # Desenhar agentes
        self._desenhar_agentes()
        
        # Adicionar informações de status
        self._adicionar_status(info_agentes)
        
        # Ajustar layout para garantir que tudo caiba
        plt.tight_layout()
        
        # Atualizar a figura
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Pausa para controlar velocidade
        if self.velocidade > 0:
            time.sleep(self.velocidade)
    
    def _desenhar_farol(self):
        """Desenha elementos específicos do ambiente Farol."""
        # Desenhar farol
        linha_f, coluna_f = self.ambiente.farol
        self._adicionar_celula(linha_f, coluna_f, 'yellow', 'orange', 0.9, linha_largura=1.5)
        self._adicionar_texto(linha_f, coluna_f, 'F', 'darkorange', 14)
        
        # Desenhar obstáculos
        for obstaculo in self.ambiente.obstaculos:
            linha_o, coluna_o = obstaculo
            self._adicionar_celula(linha_o, coluna_o, 'black', 'darkgray', 0.8)
            self._adicionar_texto(linha_o, coluna_o, 'X', 'lightgray', 12)
    
    def _desenhar_labirinto(self):
        """Desenha elementos específicos do ambiente Labirinto."""
        linhas, colunas = self.ambiente.dimensao
        
        # Desenhar paredes do labirinto
        for linha in range(linhas):
            for coluna in range(colunas):
                if self.ambiente.mapa[linha, coluna] == 1:  # Parede
                    self._adicionar_celula(linha, coluna, 'gray', 'darkgray', 0.9)
                    self._adicionar_texto(linha, coluna, '#', 'darkgray', 10)
        
        # Desenhar saída
        linha_s, coluna_s = self.ambiente.saida
        self._adicionar_celula(linha_s, coluna_s, 'green', 'darkgreen', 0.8, linha_largura=1.5)
        self._adicionar_texto(linha_s, coluna_s, 'S', 'white', 14)
        
        # Desenhar entrada
        self._adicionar_celula(0, 0, 'lightblue', 'blue', 0.6, linha_largura=1.5)
        self._adicionar_texto(0, 0, 'E', 'blue', 12)
    
    def _desenhar_agentes(self):
        """Desenha todos os agentes no ambiente."""
        cores_agentes = ['red', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
        
        for i, (agente_id, posicao) in enumerate(self.ambiente.posicoes_agentes.items()):
            if posicao is None:
                continue
                
            linha, coluna = posicao
            cor_idx = i % len(cores_agentes)
            cor = cores_agentes[cor_idx]
            
            # Desenhar agente
            self._adicionar_celula(linha, coluna, cor, 'black', 0.7, linha_largura=1.2)
            
            # Adicionar identificador (inicial do ID)
            if len(agente_id) > 0:
                inicial = agente_id[0].upper()
                self._adicionar_texto(linha, coluna, inicial, 'white', 12)
            
            # Adicionar ID completo (se houver espaço)
            linhas_amb, colunas_amb = self.ambiente.dimensao
            if colunas_amb >= 15:  # Só mostrar ID completo se o ambiente for grande o suficiente
                self._adicionar_texto(linha, coluna + 0.4, agente_id, 'black', 8)
    
    def _adicionar_status(self, info_agentes: Dict = None):
        """Adiciona informações de status à visualização."""
        linhas, colunas = self.ambiente.dimensao
        
        # Criar texto de status
        status_parts = [f"Passo: {self.passo_atual}", f"Ambiente: {self.ambiente.tipo}"]
        status_parts.append(f"Agentes: {len(self.ambiente.posicoes_agentes)}")
        
        if info_agentes:
            for agente_id, info in info_agentes.items():
                if info:
                    status_parts.append(f"{agente_id}: {info}")
        
        status_texto = " | ".join(status_parts)
        
        # Adicionar texto de status na parte superior da figura
        self.status_text = self.ax.text(
            0.5, -0.15, status_texto,
            transform=self.ax.transAxes,
            ha='center', va='top',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8)
        )
    
    def pausar(self, segundos: float = 2.0):
        """Pausa a visualização por um tempo específico."""
        plt.pause(segundos)
    
    def fechar(self):
        """Fecha a visualização."""
        plt.close(self.fig)
        plt.ioff()
        print("[Visualizador] Visualização fechada")


# Função de teste para o visualizador
def testar_visualizacao():
    """Função de teste para o visualizador."""
    print("Testando visualizador...")
    
    try:
        from ambiente import AmbienteFarol, AmbienteLabirinto
        
        # Testar visualização do Farol
        print("\n1. Criando ambiente Farol...")
        ambiente_farol = AmbienteFarol((12, 12), num_obstaculos=8)
        
        # Adicionar agentes de teste
        ambiente_farol.posicoes_agentes['Agente1'] = (0, 0)
        ambiente_farol.posicoes_agentes['Agente2'] = (5, 5)
        ambiente_farol.posicoes_agentes['Explorador'] = (11, 11)
        
        visualizador = Visualizador(ambiente_farol, "Teste - Farol", tamanho_celula=35)
        print("   Visualização aberta. Aguardando 3 segundos...")
        visualizador.pausar(3)
        
        # Simular movimento dos agentes
        print("   Simulando movimento dos agentes...")
        for passo in range(5):
            # Atualizar posições (simulação)
            ambiente_farol.posicoes_agentes['Agente1'] = (passo, passo)
            ambiente_farol.posicoes_agentes['Agente2'] = (5 + passo, 5 - passo)
            
            info_agentes = {
                'Agente1': f'Energia: {100 - passo*10}',
                'Agente2': f'Energia: {100 - passo*15}'
            }
            
            visualizador.atualizar(passo=passo, info_agentes=info_agentes)
        
        visualizador.fechar()
        
        # Testar visualização do Labirinto
        print("\n2. Criando ambiente Labirinto...")
        ambiente_lab = AmbienteLabirinto((15, 20))
        
        # Adicionar agentes de teste
        ambiente_lab.posicoes_agentes['Robo1'] = (0, 0)
        ambiente_lab.posicoes_agentes['Robo2'] = (0, 2)
        ambiente_lab.posicoes_agentes['Explorador'] = (0, 4)
        
        visualizador = Visualizador(ambiente_lab, "Teste - Labirinto", tamanho_celula=30)
        print("   Visualização aberta. Aguardando 3 segundos...")
        visualizador.pausar(3)
        
        # Simular movimento dos agentes
        print("   Simulando movimento dos agentes...")
        for passo in range(5):
            # Atualizar posições (simulação)
            ambiente_lab.posicoes_agentes['Robo1'] = (passo, passo * 2)
            ambiente_lab.posicoes_agentes['Robo2'] = (passo * 2, passo)
            
            info_agentes = {
                'Robo1': f'Dist: {passo}',
                'Robo2': f'Passos: {passo*2}'
            }
            
            visualizador.atualizar(passo=passo, info_agentes=info_agentes)
        
        visualizador.fechar()
        
        print("\nTeste de visualização concluído com sucesso!")
        
    except ImportError as e:
        print(f"\nErro de importação: {e}")
        print("Certifique-se de que o módulo 'ambiente' está disponível.")
        
        # Criar um ambiente mock para teste
        print("\nCriando ambiente mock para teste...")
        class AmbienteMock:
            def __init__(self):
                self.tipo = 'mock'
                self.dimensao = (10, 10)
                self.posicoes_agentes = {'Teste1': (5, 5), 'Teste2': (2, 8)}
        
        ambiente_mock = AmbienteMock()
        visualizador = Visualizador(ambiente_mock, "Teste - Mock", tamanho_celula=40)
        visualizador.pausar(3)
        visualizador.fechar()
        
    except Exception as e:
        print(f"\nErro durante o teste: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    testar_visualizacao()
