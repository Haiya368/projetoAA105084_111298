#!/usr/bin/env python3
"""
Módulo que define os ambientes de simulação para o problema do Farol e do Labirinto.
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Dezembro 2024
"""

import numpy as np
import random
from collections import deque
from typing import Tuple, Dict, Set, Optional, Any, List


class AmbienteBase:
    """Classe base para todos os ambientes de simulação."""
    
    def __init__(self, dimensao: Tuple[int, int], tipo: str = None):
        """
        Inicializa o ambiente base.
        
        Args:
            dimensao: Tuplo com (linhas, colunas) do ambiente
            tipo: Tipo do ambiente ('farol' ou 'labirinto')
        """
        self.dimensao = dimensao
        self.tipo = tipo
        self.posicoes_agentes: Dict[str, Tuple[int, int]] = {}
        self.terminados: Set[str] = set()
        self.historico_posicoes: Dict[str, List[Tuple[int, int]]] = {}
        self.tentativas_repetidas: Dict[str, Dict[str, int]] = {}
        self.episodios_completados = 0
        
        # Constantes de recompensa otimizadas para treino
        self.RECOMPENSA_META = 1000.0
        self.RECOMPENSA_COLISAO = -50.0
        self.RECOMPENSA_PASSO = -0.1
        self.RECOMPENSA_EXPLORACAO = 5.0
        self.RECOMPENSA_REPETICAO = -10.0
        self.RECOMPENSA_PROGRESSO = 2.0

    def reiniciar(self, posicoes_iniciais: Dict[str, Tuple[int, int]]) -> None:
        """
        Reinicia o ambiente para um novo episódio.
        
        Args:
            posicoes_iniciais: Dicionário com posições iniciais dos agentes
        """
        self.posicoes_agentes = posicoes_iniciais.copy()
        self.terminados.clear()
        self.historico_posicoes.clear()
        self.tentativas_repetidas.clear()
        self._reiniciar_estado_interno()

    def _reiniciar_estado_interno(self) -> None:
        """Método para reiniciar estado interno específico de cada ambiente."""
        pass

    def observacaoPara(self, agente_id: str) -> Optional[Dict[str, Any]]:
        """
        Retorna a observação do ambiente para um agente específico.
        
        Args:
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com informações de observação ou None se agente não existir
        """
        raise NotImplementedError("Método deve ser implementado nas subclasses")
    
    def agir(self, accao: str, agente_id: str) -> Dict[str, Any]:
        """
        Executa uma ação para um agente específico.
        
        Args:
            accao: Ação a executar (NORTE, SUL, ESTE, OESTE, PARAR)
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com resultados da ação
        """
        raise NotImplementedError("Método deve ser implementado nas subclasses")
    
    def _posicao_valida(self, linha: int, coluna: int) -> bool:
        """
        Verifica se uma posição é válida dentro dos limites do ambiente.
        
        Args:
            linha: Coordenada da linha
            coluna: Coordenada da coluna
            
        Returns:
            True se a posição for válida, False caso contrário
        """
        linhas, colunas = self.dimensao
        return 0 <= linha < linhas and 0 <= coluna < colunas
    
    def _calcular_nova_posicao(self, linha: int, coluna: int, accao: str) -> Tuple[int, int]:
        """
        Calcula nova posição com base na ação.
        
        Args:
            linha: Linha atual
            coluna: Coluna atual
            accao: Ação a executar
            
        Returns:
            Tuplo com nova posição (linha, coluna)
        """
        movimentos = {
            'NORTE': (-1, 0),
            'SUL': (1, 0),
            'ESTE': (0, 1),
            'OESTE': (0, -1),
            'PARAR': (0, 0)
        }
        if accao in movimentos:
            dl, dc = movimentos[accao]
            return linha + dl, coluna + dc
        return linha, coluna
    
    def _verificar_repeticao_posicao(self, agente_id: str, nova_posicao: Tuple[int, int]) -> float:
        """
        Verifica se o agente está repetindo posições e aplica penalização.
        
        Args:
            agente_id: Identificador do agente
            nova_posicao: Nova posição do agente
            
        Returns:
            Penalização por repetição de posição
        """
        if agente_id not in self.historico_posicoes:
            self.historico_posicoes[agente_id] = []
        
        # Adicionar posição atual ao histórico
        self.historico_posicoes[agente_id].append(nova_posicao)
        
        # Manter apenas as últimas 10 posições
        if len(self.historico_posicoes[agente_id]) > 10:
            self.historico_posicoes[agente_id] = self.historico_posicoes[agente_id][-10:]
        
        # Verificar repetições recentes
        historico_recente = self.historico_posicoes[agente_id]
        
        # Contar quantas vezes a nova posição aparece no histórico
        contagem = historico_recente.count(nova_posicao)
        
        # Penalização por repetição
        if contagem > 2:
            return self.RECOMPENSA_REPETICAO * (contagem - 1)
        return 0.0
    
    def _verificar_tentativa_repetida(self, agente_id: str, accao: str) -> float:
        """
        Verifica se o agente está tentando ações que já falharam.
        
        Args:
            agente_id: Identificador do agente
            accao: Ação a verificar
            
        Returns:
            Penalização por tentativa repetida
        """
        if agente_id not in self.tentativas_repetidas:
            self.tentativas_repetidas[agente_id] = {}
        
        if accao not in self.tentativas_repetidas[agente_id]:
            self.tentativas_repetidas[agente_id][accao] = 0
        
        # Incrementar contador para esta ação
        self.tentativas_repetidas[agente_id][accao] += 1
        
        # Penalização por tentar a mesma ação muitas vezes seguidas
        if self.tentativas_repetidas[agente_id][accao] > 5:
            return -5.0 * (self.tentativas_repetidas[agente_id][accao] - 5)
        
        return 0.0
    
    def _reset_tentativas_agente(self, agente_id: str):
        """
        Reseta contadores de tentativas para um agente.
        
        Args:
            agente_id: Identificador do agente
        """
        if agente_id in self.tentativas_repetidas:
            self.tentativas_repetidas[agente_id] = {}
    
    def episodio_completo(self, agente_id: str) -> bool:
        """
        Verifica se o episódio está completo para um agente.
        
        Args:
            agente_id: Identificador do agente
            
        Returns:
            True se o episódio estiver completo, False caso contrário
        """
        return agente_id in self.terminados
    
    def get_episodios_completados(self) -> int:
        """
        Retorna o número total de episódios completados.
        
        Returns:
            Número de episódios completados
        """
        return self.episodios_completados


class AmbienteFarol(AmbienteBase):
    """Problema do Farol com obstáculos aleatórios."""
    
    def __init__(self, dimensao: Tuple[int, int], num_obstaculos: int = 5, 
                 obstaculos_predefinidos: List[Tuple[int, int]] = None,
                 regenerar_obstaculos: bool = False):
        """
        Inicializa o ambiente Farol.
        
        Args:
            dimensao: Tuplo com (linhas, colunas)
            num_obstaculos: Número de obstáculos a gerar
            obstaculos_predefinidos: Lista de obstáculos predefinidos
            regenerar_obstaculos: Se True, regenera obstáculos a cada episódio
        """
        super().__init__(dimensao, 'farol')
        linhas, colunas = dimensao
        
        # Farol no centro
        self.farol = (linhas // 2, colunas // 2)
        self.obstaculos: Set[Tuple[int, int]] = set()
        self.num_obstaculos = num_obstaculos
        self.regenerar_obstaculos = regenerar_obstaculos
        
        if obstaculos_predefinidos:
            self.obstaculos = set(obstaculos_predefinidos)
        else:
            self._gerar_obstaculos()
        
        # Memória para reward shaping
        self.distancias_anteriores: Dict[str, float] = {}
        self.ultima_direcao: Dict[str, str] = {}
        
        print(f"[AmbienteFarol] Gerado {dimensao}")
        print(f"  Farol: {self.farol}")
        print(f"  Obstáculos: {len(self.obstaculos)}")
        print(f"  Regeneração: {'SIM' if regenerar_obstaculos else 'NÃO'}")

    def _verificar_caminho_valido(self, novo_obstaculo: Tuple[int, int]) -> bool:
        """
        Verifica se há caminho da entrada (0,0) ao farol após adicionar obstáculo.
        
        Args:
            novo_obstaculo: Posição do novo obstáculo
            
        Returns:
            True se houver caminho válido, False caso contrário
        """
        # Criar conjunto de obstáculos temporário incluindo o novo
        obstaculos_temp = self.obstaculos.copy()
        obstaculos_temp.add(novo_obstaculo)
        
        # Usar BFS para verificar se há caminho
        fila = deque([(0, 0)])
        visitados = set([(0, 0)])
        linhas, colunas = self.dimensao
        
        while fila:
            linha, coluna = fila.popleft()
            
            # Se chegou ao farol
            if (linha, coluna) == self.farol:
                return True
            
            # Verificar vizinhos
            for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nl, nc = linha + dl, coluna + dc
                
                if (0 <= nl < linhas and 0 <= nc < colunas and 
                    (nl, nc) not in obstaculos_temp and 
                    (nl, nc) not in visitados):
                    
                    visitados.add((nl, nc))
                    fila.append((nl, nc))
        
        return False

    def _gerar_obstaculos(self) -> None:
        """Gera obstáculos aleatórios garantindo que haja caminho para o farol."""
        self.obstaculos.clear()
        contador = 0
        tentativas = 0
        max_tentativas = self.num_obstaculos * 20
        
        # Lista de posições possíveis (excluindo farol e entrada)
        linhas, colunas = self.dimensao
        posicoes_possiveis = []
        for linha in range(linhas):
            for coluna in range(colunas):
                pos = (linha, coluna)
                if (pos != self.farol and pos != (0, 0) and 
                    not self._posicao_adjacente_ao_farol(pos)):
                    posicoes_possiveis.append(pos)
        
        # Embaralhar posições possíveis
        random.shuffle(posicoes_possiveis)
        
        # Tentar adicionar obstáculos que não bloqueiem o caminho
        for posicao in posicoes_possiveis:
            if contador >= self.num_obstaculos:
                break
                
            # Verificar se adicionar este obstáculo bloqueia o caminho
            if self._verificar_caminho_valido(posicao):
                self.obstaculos.add(posicao)
                contador += 1
        
        # Se não conseguiu todos os obstáculos, completar com posições que não bloqueiam
        if contador < self.num_obstaculos:
            print(f"[AmbienteFarol] Apenas {contador} obstáculos gerados com caminho válido.")
            # Tentar posições restantes
            for posicao in posicoes_possiveis:
                if contador >= self.num_obstaculos:
                    break
                if posicao not in self.obstaculos:
                    self.obstaculos.add(posicao)
                    contador += 1
        
        print(f"[AmbienteFarol] Obstáculos gerados: {len(self.obstaculos)}/{self.num_obstaculos}")

    def _posicao_adjacente_ao_farol(self, posicao: Tuple[int, int]) -> bool:
        """
        Verifica se a posição é adjacente ao farol.
        
        Args:
            posicao: Posição a verificar
            
        Returns:
            True se a posição for adjacente ao farol, False caso contrário
        """
        linha, coluna = posicao
        linha_f, coluna_f = self.farol
        return abs(linha - linha_f) <= 1 and abs(coluna - coluna_f) <= 1

    def _reiniciar_estado_interno(self) -> None:
        """Reinicia o estado interno do ambiente Farol."""
        self.distancias_anteriores.clear()
        self.ultima_direcao.clear()
        if self.regenerar_obstaculos:
            self._gerar_obstaculos()

    def _calcular_direcao_farol(self, posicao: Tuple[int, int]) -> str:
        """
        Calcula a direção do farol em relação à posição.
        
        Args:
            posicao: Posição atual
            
        Returns:
            Direção do farol (NORTE, SUL, ESTE, OESTE)
        """
        linha_agente, coluna_agente = posicao
        linha_farol, coluna_farol = self.farol
        diferenca_linha = linha_farol - linha_agente
        diferenca_coluna = coluna_farol - coluna_agente
        
        if abs(diferenca_linha) >= abs(diferenca_coluna):
            return "SUL" if diferenca_linha > 0 else "NORTE"
        return "ESTE" if diferenca_coluna > 0 else "OESTE"

    def observacaoPara(self, agente_id: str) -> Optional[Dict[str, Any]]:
        """
        Retorna observação para um agente no ambiente Farol.
        
        Args:
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com informações de observação
        """
        if agente_id in self.terminados:
            return {'TERMINOU': True}
        
        posicao = self.posicoes_agentes.get(agente_id)
        if posicao is None:
            return None
        
        # Calcular distância ao farol
        distancia = abs(posicao[0] - self.farol[0]) + abs(posicao[1] - self.farol[1])
        
        # Calcular direções livres
        direcoes_livres = []
        for direcao, offset in [('NORTE', (-1, 0)), ('SUL', (1, 0)), ('ESTE', (0, 1)), ('OESTE', (0, -1))]:
            nl, nc = posicao[0] + offset[0], posicao[1] + offset[1]
            if self._posicao_valida(nl, nc) and (nl, nc) not in self.obstaculos:
                direcoes_livres.append(direcao)
        
        return {
            'posicao': posicao,
            'direcao_farol': self._calcular_direcao_farol(posicao),
            'direcoes_livres': direcoes_livres,
            'distancia_farol': distancia,
            'mapa_ref': self
        }

    def agir(self, accao: str, agente_id: str) -> Dict[str, Any]:
        """
        Executa ação para um agente no ambiente Farol.
        
        Args:
            accao: Ação a executar
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com resultados da ação
        """
        if agente_id in self.terminados:
            return {
                'recompensa': 0.0,
                'terminou': True,
                'posicao_nova': self.posicoes_agentes[agente_id],
                'episodio_completo': True
            }
        
        linha, coluna = self.posicoes_agentes[agente_id]
        nova_linha, nova_coluna = self._calcular_nova_posicao(linha, coluna, accao)
        
        # Verificar colisões
        colisao = False
        if not self._posicao_valida(nova_linha, nova_coluna):
            colisao = True
        elif (nova_linha, nova_coluna) in self.obstaculos:
            colisao = True
        
        if colisao:
            # Penalização por colisão
            recompensa_colisao = self.RECOMPENSA_COLISAO
            
            # Penalização adicional por tentativa repetida
            recompensa_colisao += self._verificar_tentativa_repetida(agente_id, accao)
            
            return {
                'recompensa': recompensa_colisao,
                'terminou': False,
                'posicao_nova': (linha, coluna),
                'colisao': True,
                'episodio_completo': False
            }
        
        # Movimento válido - resetar tentativas para esta ação
        if agente_id in self.tentativas_repetidas and accao in self.tentativas_repetidas[agente_id]:
            self.tentativas_repetidas[agente_id][accao] = 0
        
        self.posicoes_agentes[agente_id] = (nova_linha, nova_coluna)
        distancia_atual = abs(nova_linha - self.farol[0]) + abs(nova_coluna - self.farol[1])
        
        # Verificar se alcançou o farol
        if (nova_linha, nova_coluna) == self.farol:
            recompensa = self.RECOMPENSA_META
            terminou = True
            self.terminados.add(agente_id)
            self.episodios_completados += 1
            print(f"[AmbienteFarol] Agente {agente_id} alcançou o farol! Episódios completados: {self.episodios_completados}")
        else:
            recompensa = self.RECOMPENSA_PASSO
            terminou = False
            
            # Reward shaping (incentivo por aproximação)
            distancia_anterior = self.distancias_anteriores.get(agente_id, distancia_atual)
            if distancia_atual < distancia_anterior:
                recompensa += self.RECOMPENSA_PROGRESSO * 2
            elif distancia_atual > distancia_anterior:
                recompensa -= self.RECOMPENSA_PROGRESSO
            
            self.distancias_anteriores[agente_id] = distancia_atual
            
            # Penalização por repetição de posição
            recompensa += self._verificar_repeticao_posicao(agente_id, (nova_linha, nova_coluna))
            
            # Bónus por exploração (quando se move para nova área)
            if self._e_nova_area(agente_id, (nova_linha, nova_coluna)):
                recompensa += self.RECOMPENSA_EXPLORACAO

        return {
            'recompensa': recompensa,
            'terminou': terminou,
            'posicao_nova': (nova_linha, nova_coluna),
            'colisao': False,
            'episodio_completo': terminou
        }
    
    def _e_nova_area(self, agente_id: str, posicao: Tuple[int, int]) -> bool:
        """
        Verifica se a posição é uma nova área para o agente.
        
        Args:
            agente_id: Identificador do agente
            posicao: Posição a verificar
            
        Returns:
            True se for uma nova área, False caso contrário
        """
        if agente_id not in self.historico_posicoes:
            return True
        
        # Considerar nova área se não esteve nas últimas 5 posições
        ultimas_posicoes = self.historico_posicoes[agente_id][-5:] if len(self.historico_posicoes[agente_id]) >= 5 else self.historico_posicoes[agente_id]
        return posicao not in ultimas_posicoes


class AmbienteLabirinto(AmbienteBase):
    """Problema do Labirinto com geração procedural."""
    
    def __init__(self, dimensao: Tuple[int, int], 
                 mapa_predefinido: np.ndarray = None,
                 regenerar_labirinto: bool = False):
        """
        Inicializa o ambiente Labirinto.
        
        Args:
            dimensao: Tuplo com (linhas, colunas)
            mapa_predefinido: Mapa predefinido (opcional)
            regenerar_labirinto: Se True, regenera labirinto a cada episódio
        """
        super().__init__(dimensao, 'labirinto')
        self.saida = (dimensao[0] - 1, dimensao[1] - 1)
        self.regenerar_labirinto = regenerar_labirinto
        
        if mapa_predefinido is not None:
            self.mapa = mapa_predefinido.copy()
        else:
            self.mapa = np.ones(dimensao, dtype=int)
            self._gerar_labirinto_procedural()
        
        self.celulas_visitadas: Dict[str, Set[Tuple[int, int]]] = {}
        self.ultimas_direcoes: Dict[str, List[str]] = {}
        
        print(f"[AmbienteLabirinto] Labirinto {dimensao} gerado.")
        print(f"  Saída: {self.saida}")
        print(f"  Regeneração: {'SIM' if regenerar_labirinto else 'NÃO'}")

    def _gerar_labirinto_procedural(self) -> None:
        """Gera um labirinto perfeito usando algoritmo de Recursive Backtracking."""
        linhas, colunas = self.dimensao
        
        # Inicializar tudo como paredes
        self.mapa.fill(1)
        
        # Ponto de partida
        inicio_linha, inicio_coluna = 0, 0
        self.mapa[inicio_linha, inicio_coluna] = 0
        
        pilha = [(inicio_linha, inicio_coluna)]
        
        while pilha:
            linha, coluna = pilha[-1]
            vizinhos = []
            
            # Verificar vizinhos a 2 células de distância
            for dl, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nl, nc = linha + dl, coluna + dc
                if 0 <= nl < linhas and 0 <= nc < colunas and self.mapa[nl, nc] == 1:
                    vizinhos.append((nl, nc, dl//2, dc//2))
            
            if vizinhos:
                nl, nc, wl, wc = random.choice(vizinhos)
                self.mapa[linha + wl, coluna + wc] = 0
                self.mapa[nl, nc] = 0
                pilha.append((nl, nc))
            else:
                pilha.pop()
        
        # Garantir que entrada e saída estão livres
        self.mapa[0, 0] = 0
        self.mapa[self.saida] = 0
        
        # Garantir que há caminho até à saída
        self._garantir_caminho_saida()

    def _garantir_caminho_saida(self):
        """Garante que existe um caminho da entrada à saída."""
        linhas, colunas = self.dimensao
        
        # Usar BFS para verificar conectividade
        visitados = set()
        fila = deque([(0, 0)])
        
        while fila:
            linha, coluna = fila.popleft()
            if (linha, coluna) in visitados:
                continue
            visitados.add((linha, coluna))
            
            if (linha, coluna) == self.saida:
                return  # Já existe caminho
            
            for dl, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nl, nc = linha + dl, coluna + dc
                if (0 <= nl < linhas and 0 <= nc < colunas and 
                    self.mapa[nl, nc] == 0 and (nl, nc) not in visitados):
                    fila.append((nl, nc))
        
        # Se não chegou à saída, criar um caminho
        print("[AmbienteLabirinto] Criando caminho para a saída...")
        # Algoritmo simples: ir descendo e para a direita
        linha, coluna = 0, 0
        while (linha, coluna) != self.saida:
            if linha < self.saida[0]:
                if self.mapa[linha + 1, coluna] == 1:
                    self.mapa[linha + 1, coluna] = 0
                linha += 1
            if coluna < self.saida[1]:
                if self.mapa[linha, coluna + 1] == 1:
                    self.mapa[linha, coluna + 1] = 0
                coluna += 1

    def _reiniciar_estado_interno(self) -> None:
        """Reinicia o estado interno do ambiente Labirinto."""
        self.celulas_visitadas.clear()
        self.ultimas_direcoes.clear()
        if self.regenerar_labirinto:
            self._gerar_labirinto_procedural()

    def observacaoPara(self, agente_id: str) -> Optional[Dict[str, Any]]:
        """
        Retorna observação para um agente no ambiente Labirinto.
        
        Args:
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com informações de observação
        """
        if agente_id in self.terminados:
            return {'TERMINOU': True}
        
        posicao = self.posicoes_agentes.get(agente_id)
        if posicao is None:
            return None
        
        linha, coluna = posicao
        
        # Calcular visão 3x3
        visao = {}
        direcoes_livres = []
        
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nova_linha, nova_coluna = linha + dl, coluna + dc
                
                if self._posicao_valida(nova_linha, nova_coluna):
                    if self.mapa[nova_linha, nova_coluna] == 1:
                        visao[(dl, dc)] = 'PAREDE'
                    else:
                        visao[(dl, dc)] = 'LIVRE'
                        # Verificar se é direção cardinal livre
                        if abs(dl) + abs(dc) == 1:  # NORTE, SUL, ESTE, OESTE
                            direcoes_livres.append(self._offset_para_direcao(dl, dc))
                else:
                    visao[(dl, dc)] = 'PAREDE'
        
        # Calcular distância à saída
        distancia = abs(linha - self.saida[0]) + abs(coluna - self.saida[1])
        
        return {
            'posicao': posicao,
            'visao': visao,
            'direcoes_livres': direcoes_livres,
            'distancia_saida': distancia,
            'mapa_ref': self
        }
    
    def _offset_para_direcao(self, dl: int, dc: int) -> str:
        """
        Converte offset para direção.
        
        Args:
            dl: Deslocamento vertical
            dc: Deslocamento horizontal
            
        Returns:
            Direção correspondente
        """
        if dl == -1 and dc == 0:
            return 'NORTE'
        elif dl == 1 and dc == 0:
            return 'SUL'
        elif dl == 0 and dc == 1:
            return 'ESTE'
        elif dl == 0 and dc == -1:
            return 'OESTE'
        return 'PARAR'

    def agir(self, accao: str, agente_id: str) -> Dict[str, Any]:
        """
        Executa ação para um agente no ambiente Labirinto.
        
        Args:
            accao: Ação a executar
            agente_id: Identificador do agente
            
        Returns:
            Dicionário com resultados da ação
        """
        if agente_id in self.terminados:
            return {
                'recompensa': 0.0,
                'terminou': True,
                'posicao_nova': self.posicoes_agentes[agente_id],
                'episodio_completo': True
            }

        linha, coluna = self.posicoes_agentes[agente_id]
        nova_linha, nova_coluna = self._calcular_nova_posicao(linha, coluna, accao)
        
        # Inicializar registo de exploração
        if agente_id not in self.celulas_visitadas:
            self.celulas_visitadas[agente_id] = set()
            self.celulas_visitadas[agente_id].add((linha, coluna))
        
        # Inicializar histórico de direções
        if agente_id not in self.ultimas_direcoes:
            self.ultimas_direcoes[agente_id] = []
        
        # Verificar colisão
        colisao = False
        if not self._posicao_valida(nova_linha, nova_coluna):
            colisao = True
        elif self.mapa[nova_linha, nova_coluna] == 1:
            colisao = True
        
        if colisao:
            # Penalização por colisão
            recompensa_colisao = self.RECOMPENSA_COLISAO
            
            # Penalização adicional por tentativa repetida
            recompensa_colisao += self._verificar_tentativa_repetida(agente_id, accao)
            
            # Penalização por voltar atrás repetidamente
            if len(self.ultimas_direcoes[agente_id]) >= 2:
                ultima_dir = self.ultimas_direcoes[agente_id][-1] if self.ultimas_direcoes[agente_id] else ''
                if accao == self._direcao_oposta(ultima_dir):
                    recompensa_colisao -= 2.0
            
            return {
                'recompensa': recompensa_colisao,
                'terminou': False,
                'posicao_nova': (linha, coluna),
                'colisao': True,
                'episodio_completo': False
            }
        
        # Movimento válido - resetar tentativas para esta ação
        if agente_id in self.tentativas_repetidas and accao in self.tentativas_repetidas[agente_id]:
            self.tentativas_repetidas[agente_id][accao] = 0
        
        # Atualizar posição
        self.posicoes_agentes[agente_id] = (nova_linha, nova_coluna)
        
        # Atualizar histórico de direções
        self.ultimas_direcoes[agente_id].append(accao)
        if len(self.ultimas_direcoes[agente_id]) > 5:
            self.ultimas_direcoes[agente_id] = self.ultimas_direcoes[agente_id][-5:]
        
        # Calcular recompensa base
        recompensa = self.RECOMPENSA_PASSO
        
        # Bónus de exploração
        celula_nova = (nova_linha, nova_coluna)
        if celula_nova not in self.celulas_visitadas[agente_id]:
            recompensa += self.RECOMPENSA_EXPLORACAO * 3
            self.celulas_visitadas[agente_id].add(celula_nova)
        
        # Penalização por repetição de posição
        recompensa += self._verificar_repeticao_posicao(agente_id, (nova_linha, nova_coluna))
        
        # Recompensa por progresso em direção à saída
        distancia_antiga = abs(linha - self.saida[0]) + abs(coluna - self.saida[1])
        distancia_nova = abs(nova_linha - self.saida[0]) + abs(nova_coluna - self.saida[1])
        
        if distancia_nova < distancia_antiga:
            recompensa += self.RECOMPENSA_PROGRESSO * 2
        elif distancia_nova > distancia_antiga:
            recompensa -= self.RECOMPENSA_PROGRESSO
        
        # Penalização por voltar atrás rapidamente
        if len(self.ultimas_direcoes[agente_id]) >= 2:
            if self.ultimas_direcoes[agente_id][-1] == self._direcao_oposta(self.ultimas_direcoes[agente_id][-2]):
                recompensa -= 1.5
        
        terminou = False
        if (nova_linha, nova_coluna) == self.saida:
            recompensa += self.RECOMPENSA_META
            terminou = True
            self.terminados.add(agente_id)
            self.episodios_completados += 1
            print(f"[AmbienteLabirinto] Agente {agente_id} encontrou a saída! Episódios completados: {self.episodios_completados}")
        
        return {
            'recompensa': recompensa,
            'terminou': terminou,
            'posicao_nova': (nova_linha, nova_coluna),
            'colisao': False,
            'episodio_completo': terminou
        }
    
    def _direcao_oposta(self, direcao: str) -> str:
        """
        Retorna a direção oposta.
        
        Args:
            direcao: Direção atual
            
        Returns:
            Direção oposta
        """
        opostas = {
            'NORTE': 'SUL',
            'SUL': 'NORTE',
            'ESTE': 'OESTE',
            'OESTE': 'ESTE'
        }
        return opostas.get(direcao, direcao)