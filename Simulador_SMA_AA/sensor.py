#!/usr/bin/env python3
"""
Módulo que define os sensores para os agentes.
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Dezembro 2024
"""

from typing import Dict, Any, Tuple, List


class SensorBase:
    """Classe base para todos os sensores."""
    
    def __init__(self):
        """
        Inicializa o sensor base.
        """
        self.tipo = "base"
    
    def observar(self, ambiente: Any, posicao_agente: Tuple[int, int]) -> Dict[str, Any]:
        """
        Observa o ambiente a partir da posição do agente.
        
        Args:
            ambiente: Referência ao ambiente
            posicao_agente: Posição atual do agente
            
        Returns:
            Dicionário com informações observadas
        """
        raise NotImplementedError("Método observar deve ser implementado nas subclasses")


class SensorDirecional(SensorBase):
    """Sensor para o problema do Farol - deteta direção do farol."""
    
    def __init__(self):
        """
        Inicializa o sensor direcional.
        """
        super().__init__()
        self.tipo = "direcional"
    
    def observar(self, ambiente: Any, posicao_agente: Tuple[int, int]) -> Dict[str, Any]:
        """
        Observa a direção do farol e obstáculos próximos.
        
        Args:
            ambiente: Ambiente Farol
            posicao_agente: Posição atual do agente
            
        Returns:
            Dicionário com direção do farol e direções livres
        """
        if not hasattr(ambiente, 'farol'):
            return {'direcao_farol': 'DESCONHECIDA'}
        
        linha_agente, coluna_agente = posicao_agente
        linha_farol, coluna_farol = ambiente.farol
        
        # Calcular direção do farol
        diferenca_linha = linha_farol - linha_agente
        diferenca_coluna = coluna_farol - coluna_agente
        
        if abs(diferenca_linha) >= abs(diferenca_coluna):
            direcao = "SUL" if diferenca_linha > 0 else "NORTE"
        else:
            direcao = "ESTE" if diferenca_coluna > 0 else "OESTE"
        
        # Detetar obstáculos nas 4 direções cardinais
        direcoes_livres = []
        for direcao_teste, offset in [('NORTE', (-1, 0)), ('SUL', (1, 0)), ('ESTE', (0, 1)), ('OESTE', (0, -1))]:
            nl, nc = linha_agente + offset[0], coluna_agente + offset[1]
            
            # Verificar se a posição é válida e não tem obstáculo
            if ambiente._posicao_valida(nl, nc):
                if hasattr(ambiente, 'obstaculos'):
                    if (nl, nc) not in ambiente.obstaculos:
                        direcoes_livres.append(direcao_teste)
                elif hasattr(ambiente, 'mapa'):
                    if ambiente.mapa[nl, nc] == 0:
                        direcoes_livres.append(direcao_teste)
                else:
                    direcoes_livres.append(direcao_teste)
            else:
                # Posição inválida (fora dos limites) é tratada como obstáculo
                pass
        
        return {
            'direcao_farol': direcao,
            'direcoes_livres': direcoes_livres,
            'distancia': abs(diferenca_linha) + abs(diferenca_coluna)
        }


class SensorVisao3x3(SensorBase):
    """Sensor para o problema do Labirinto - visão 3x3 ao redor do agente."""
    
    def __init__(self):
        """
        Inicializa o sensor de visão 3x3.
        """
        super().__init__()
        self.tipo = "visao_3x3"
    
    def observar(self, ambiente: Any, posicao_agente: Tuple[int, int]) -> Dict[str, Any]:
        """
        Observa o ambiente em uma área 3x3 ao redor do agente.
        
        Args:
            ambiente: Ambiente Labirinto
            posicao_agente: Posição atual do agente
            
        Returns:
            Dicionário com visão do ambiente
        """
        linha_agente, coluna_agente = posicao_agente
        
        # Inicializar visão 3x3
        visao = {}
        direcoes_livres = []
        
        # Varrer área 3x3 ao redor do agente
        for dl in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                nova_linha = linha_agente + dl
                nova_coluna = coluna_agente + dc
                
                # Verificar se a posição é válida
                if ambiente._posicao_valida(nova_linha, nova_coluna):
                    # Verificar se é parede ou caminho livre
                    if hasattr(ambiente, 'mapa'):
                        if ambiente.mapa[nova_linha, nova_coluna] == 1:
                            visao[(dl, dc)] = 'PAREDE'
                        else:
                            visao[(dl, dc)] = 'LIVRE'
                            
                            # Se for direção cardinal (norte, sul, este, oeste), adicionar às direções livres
                            if abs(dl) + abs(dc) == 1:
                                direcao = self._offset_para_direcao(dl, dc)
                                direcoes_livres.append(direcao)
                    else:
                        visao[(dl, dc)] = 'DESCONHECIDO'
                else:
                    # Posição fora dos limites é tratada como parede
                    visao[(dl, dc)] = 'PAREDE'
        
        # Calcular direção da saída se disponível
        distancia_saida = None
        if hasattr(ambiente, 'saida'):
            distancia_saida = abs(linha_agente - ambiente.saida[0]) + abs(coluna_agente - ambiente.saida[1])
        
        return {
            'visao': visao,
            'direcoes_livres': direcoes_livres,
            'distancia_saida': distancia_saida
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