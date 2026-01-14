#!/usr/bin/env python3
"""
Módulo que define os agentes para o simulador SMA.
Versão completa com suporte a treino de 10.000 episódios
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Janeiro 2026
"""

import os
import pickle
import numpy as np
import random
import threading
from collections import deque, defaultdict
from typing import Tuple, Dict, Any, List, Optional, Set
import time


class AgenteBase:
    """Classe base para todos os agentes."""
    
    def __init__(self, id_agente: str, posicao_inicial: Tuple[int, int]):
        """
        Inicializa o agente base.
        
        Args:
            id_agente: Identificador único do agente
            posicao_inicial: Tuplo com posição inicial (linha, coluna)
        """
        self.id = id_agente
        self.posicao = posicao_inicial
        self.sensor = None
        self.percepcao = None
        self.accoes = ['NORTE', 'SUL', 'ESTE', 'OESTE', 'PARAR']
        self.terminado = False
        self.data_criacao = time.time()
        self.ultima_atualizacao = time.time()
    
    def instala(self, sensor) -> None:
        """
        Instala um sensor no agente.
        
        Args:
            sensor: Sensor a instalar
        """
        self.sensor = sensor
    
    def observacao(self, percepcao: Dict[str, Any]) -> None:
        """
        Recebe uma observação do ambiente.
        
        Args:
            percepcao: Dicionário com informações de percepção
        """
        self.percepcao = percepcao
        self.ultima_atualizacao = time.time()
    
    def age(self, modo: str = 'teste') -> str:
        """
        Escolhe uma ação usando regras fixas melhoradas.
        
        Args:
            modo: Modo de operação (ignorado para agente fixo)
            
        Returns:
            Ação escolhida
        """
        if self.percepcao is None:
            return 'PARAR'
        
        # Se o episódio terminou
        if self.percepcao.get('TERMINOU', False):
            self.terminado = True
            return 'PARAR'
        
        # Lógica para Farol: seguir direção do farol se livre
        if 'direcao_farol' in self.percepcao:
            direcao_obj = self.percepcao['direcao_farol']
            direcoes_livres = self.percepcao.get('direcoes_livres', [])
            
            # Prioridade 1: Ir na direção do farol se livre
            if direcao_obj in direcoes_livres:
                return direcao_obj
            
            # Prioridade 2: Tentar direções que aproximem do farol
            if direcoes_livres:
                # Calcular qual direção livre mais aproxima do farol
                melhor_direcao = None
                melhor_melhoria = -float('inf')
                
                for direcao in direcoes_livres:
                    # Simular movimento
                    nova_linha, nova_coluna = self.posicao
                    if direcao == 'NORTE':
                        nova_linha -= 1
                    elif direcao == 'SUL':
                        nova_linha += 1
                    elif direcao == 'ESTE':
                        nova_coluna += 1
                    elif direcao == 'OESTE':
                        nova_coluna -= 1
                    
                    # Calcular se melhora direção para farol
                    if 'mapa_ref' in self.percepcao:
                        ambiente = self.percepcao['mapa_ref']
                        if hasattr(ambiente, 'farol'):
                            linha_f, coluna_f = ambiente.farol
                            distancia_atual = abs(self.posicao[0] - linha_f) + abs(self.posicao[1] - coluna_f)
                            distancia_nova = abs(nova_linha - linha_f) + abs(nova_coluna - coluna_f)
                            
                            if distancia_nova < distancia_atual:
                                melhoria = distancia_atual - distancia_nova
                                if melhoria > melhor_melhoria:
                                    melhor_melhoria = melhoria
                                    melhor_direcao = direcao
                
                if melhor_direcao:
                    return melhor_direcao
                
                # Prioridade 3: Direção aleatória se não houver melhoria
                return random.choice(direcoes_livres)
        
        # Lógica para Labirinto: algoritmo da mão direita (seguir parede direita)
        elif 'visao' in self.percepcao:
            direcoes_livres = self.percepcao.get('direcoes_livres', [])
            
            if not direcoes_livres:
                return 'PARAR'
            
            # Ordem de preferência: ESTE, SUL, OESTE, NORTE (algoritmo mão direita)
            for direcao in ['ESTE', 'SUL', 'OESTE', 'NORTE']:
                if direcao in direcoes_livres:
                    return direcao
            
            # Fallback
            return random.choice(direcoes_livres)
        
        # Fallback geral
        return 'PARAR'
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """
        Avalia o estado atual com base na recompensa recebida.
        
        Args:
            recompensa: Recompensa recebida
        """
        pass
    
    def reiniciar(self) -> None:
        """Reinicia o agente para um novo episódio."""
        self.percepcao = None
        self.terminado = False
    
    def obter_estatisticas_aprendizagem(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de aprendizagem do agente.
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            'estados_aprendidos': 0,
            'epsilon_atual': 0.0,
            'taxa_exploracao': 0.0,
            'exploracoes_total': 0
        }


class AgenteFixo(AgenteBase):
    """Agente com comportamento fixo (regras pré-definidas)."""
    
    def __init__(self, id_agente: str, posicao_inicial: Tuple[int, int]):
        """
        Inicializa o agente fixo.
        
        Args:
            id_agente: Identificador do agente
            posicao_inicial: Posição inicial
        """
        super().__init__(id_agente, posicao_inicial)
        self.tipo = "fixo"
        self.colisoes = 0
        self.passos = 0
        self.recompensa_acumulada = 0.0
    
    def age(self, modo: str = 'teste') -> str:
        """
        Escolhe uma ação usando regras fixas.
        
        Args:
            modo: Modo de operação (ignorado para agente fixo)
            
        Returns:
            Ação escolhida
        """
        if self.percepcao is None:
            return 'PARAR'
        
        # Se o episódio terminou
        if self.percepcao.get('TERMINOU', False):
            self.terminado = True
            return 'PARAR'
        
        # Lógica para Farol: seguir direção do farol se livre
        if 'direcao_farol' in self.percepcao:
            direcao_obj = self.percepcao['direcao_farol']
            direcoes_livres = self.percepcao.get('direcoes_livres', [])
            
            # Tentar ir na direção do farol
            if direcao_obj in direcoes_livres:
                return direcao_obj
            
            # Se não for possível, tentar outra direção livre
            if direcoes_livres:
                return random.choice(direcoes_livres)
        
        # Lógica para Labirinto: seguir parede direita (algoritmo simples)
        elif 'visao' in self.percepcao:
            direcoes_livres = self.percepcao.get('direcoes_livres', [])
            
            if not direcoes_livres:
                return 'PARAR'
            
            # Prioridade: ESTE, SUL, OESTE, NORTE (parede direita)
            for direcao in ['ESTE', 'SUL', 'OESTE', 'NORTE']:
                if direcao in direcoes_livres:
                    return direcao
            
            # Fallback
            return random.choice(direcoes_livres)
        
        # Fallback geral
        return 'PARAR'
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """
        Atualiza estatísticas do agente.
        
        Args:
            recompensa: Recompensa recebida
        """
        self.recompensa_acumulada += recompensa
        self.passos += 1
        
        if recompensa < -10:  # Penalização forte indica colisão
            self.colisoes += 1


class AgenteRL(AgenteBase):
    """Agente com Q-Learning básico."""
    
    def __init__(self, id_agente: str, posicao_inicial: Tuple[int, int],
                 alfa: float = 0.1, gama: float = 0.9, epsilon: float = 0.9,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.05):
        """
        Inicializa o agente RL.
        
        Args:
            id_agente: Identificador do agente
            posicao_inicial: Posição inicial
            alfa: Taxa de aprendizagem
            gama: Fator de desconto
            epsilon: Taxa de exploração inicial
            epsilon_decay: Taxa de decaimento do epsilon
            epsilon_min: Valor mínimo do epsilon
        """
        super().__init__(id_agente, posicao_inicial)
        self.tipo = "rl"
        
        # Parâmetros de aprendizagem
        self.alfa = alfa
        self.gama = gama
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Memória
        self.q_tabela = {}
        self.estado_anterior = None
        self.accao_anterior = None
        
        # Estatísticas
        self.episodios_treinados = 0
        self.exploracoes_total = 0
        self.exploracoes = 0
        self.melhor_recompensa = -float('inf')
        self.melhor_episodio = 0
        self.recompensa_episodio = 0.0
        self.passos_episodio = 0
    
    def _extrair_estado(self) -> str:
        """
        Extrai uma representação do estado a partir da percepção.
        
        Returns:
            String representando o estado
        """
        if self.percepcao is None:
            return "INICIO"
        
        # Se o episódio terminou
        if self.percepcao.get('TERMINOU', False):
            return "TERMINOU"
        
        estado_parts = []
        
        # Para ambiente farol
        if 'direcao_farol' in self.percepcao:
            estado_parts.append(f"DIR_{self.percepcao['direcao_farol']}")
            estado_parts.append(f"DIST_{self.percepcao.get('distancia_farol', 0)}")
            
            # Direções livres
            dir_livres = self.percepcao.get('direcoes_livres', [])
            for dir in ['NORTE', 'SUL', 'ESTE', 'OESTE']:
                estado_parts.append(f"{dir}_{'L' if dir in dir_livres else 'B'}")
        
        # Para ambiente labirinto
        elif 'visao' in self.percepcao:
            visao = self.percepcao['visao']
            
            # Visão 3x3 simplificada
            for (dl, dc), valor in visao.items():
                if abs(dl) <= 1 and abs(dc) <= 1:
                    estado_parts.append(f"{dl},{dc}_{valor[0]}")
            
            # Direções livres
            dir_livres = self.percepcao.get('direcoes_livres', [])
            for dir in ['NORTE', 'SUL', 'ESTE', 'OESTE']:
                estado_parts.append(f"{dir}_{'L' if dir in dir_livres else 'B'}")
            
            # Distância à saída
            if 'distancia_saida' in self.percepcao:
                estado_parts.append(f"DIST_{self.percepcao['distancia_saida']}")
        
        return "|".join(estado_parts)
    
    def age(self, modo: str = 'teste') -> str:
        """
        Escolhe uma ação usando Q-Learning.
        
        Args:
            modo: Modo de operação
            
        Returns:
            Ação escolhida
        """
        if self.percepcao is None:
            return 'PARAR'
        
        # Se o episódio terminou
        if self.percepcao.get('TERMINOU', False):
            self.terminado = True
            return 'PARAR'
        
        estado = self._extrair_estado()
        
        # Inicializar Q-values para este estado se necessário
        if estado not in self.q_tabela:
            self.q_tabela[estado] = {accao: 0.0 for accao in self.accoes}
        
        # Escolher ação (ε-greedy)
        if modo == 'aprendizagem' and random.random() < self.epsilon:
            # Exploração: escolher ação aleatória (exceto PARAR)
            accoes_validas = [a for a in self.accoes if a != 'PARAR']
            accao = random.choice(accoes_validas)
            self.exploracoes += 1
            self.exploracoes_total += 1
        else:
            # Exploração: escolher melhor ação
            q_valores = self.q_tabela[estado]
            
            # Filtrar ações válidas (não PARAR)
            q_valores_validos = {a: v for a, v in q_valores.items() if a != 'PARAR'}
            
            if q_valores_validos:
                # Encontrar valor máximo
                max_valor = max(q_valores_validos.values())
                
                # Escolher entre ações com valor máximo
                melhores_accoes = [a for a, v in q_valores_validos.items() if v == max_valor]
                accao = random.choice(melhores_accoes)
            else:
                # Fallback
                accao = random.choice([a for a in self.accoes if a != 'PARAR'])
        
        # Guardar para aprendizagem
        if modo == 'aprendizagem':
            self.estado_anterior = estado
            self.accao_anterior = accao
        
        return accao
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """
        Atualiza Q-values usando Q-Learning.
        
        Args:
            recompensa: Recompensa recebida
        """
        self.recompensa_episodio += recompensa
        self.passos_episodio += 1
        
        if self.estado_anterior is None or self.accao_anterior is None:
            return
        
        estado_atual = self._extrair_estado()
        
        # Garantir que o estado atual está na Q-table
        if estado_atual not in self.q_tabela:
            self.q_tabela[estado_atual] = {accao: 0.0 for accao in self.accoes}
        
        # Q-Learning update
        q_antigo = self.q_tabela[self.estado_anterior][self.accao_anterior]
        
        # Encontrar valor máximo para o estado atual
        max_q_futuro = max(self.q_tabela[estado_atual].values())
        
        # Calcular novo valor Q
        novo_q = q_antigo + self.alfa * (recompensa + self.gama * max_q_futuro - q_antigo)
        
        # Atualizar Q-value
        self.q_tabela[self.estado_anterior][self.accao_anterior] = novo_q
        
        # Resetar para próximo passo
        self.estado_anterior = None
        self.accao_anterior = None
    
    def atualizar_melhor(self, episodio: int) -> None:
        """
        Atualiza estatísticas do melhor agente.
        
        Args:
            episodio: Número do episódio atual
        """
        if self.recompensa_episodio > self.melhor_recompensa:
            self.melhor_recompensa = self.recompensa_episodio
            self.melhor_episodio = episodio
        
        # Resetar para próximo episódio
        self.episodios_treinados += 1
        self.recompensa_episodio = 0.0
        self.passos_episodio = 0
        
        # Decaimento do epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def reiniciar(self) -> None:
        """
        Reinicia o agente para um novo episódio.
        """
        super().reiniciar()
        self.estado_anterior = None
        self.accao_anterior = None
        self.recompensa_episodio = 0.0
        self.passos_episodio = 0
        self.exploracoes = 0
    
    def obter_estatisticas_aprendizagem(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de aprendizagem.
        
        Returns:
            Dicionário com estatísticas
        """
        taxa_exploracao = 0.0
        if self.passos_episodio > 0:
            taxa_exploracao = self.exploracoes / self.passos_episodio
        
        return {
            'estados_aprendidos': len(self.q_tabela),
            'epsilon_atual': self.epsilon,
            'taxa_exploracao': taxa_exploracao,
            'exploracoes_total': self.exploracoes_total
        }
    
    def salvar_agente(self, filename: str) -> None:
        """
        Salva o agente em um arquivo.
        
        Args:
            filename: Nome do arquivo
        """
        dados = {
            'id': self.id,
            'tipo': self.tipo,
            'q_tabela': self.q_tabela,
            'epsilon': self.epsilon,
            'melhor_recompensa': self.melhor_recompensa,
            'melhor_episodio': self.melhor_episodio,
            'episodios_treinados': self.episodios_treinados,
            'exploracoes_total': self.exploracoes_total,
            'parametros': {
                'alfa': self.alfa,
                'gama': self.gama,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            },
            'data_criacao': self.data_criacao,
            'ultima_atualizacao': time.time()
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(dados, f)
    
    def carregar_agente(self, filename: str) -> None:
        """
        Carrega o agente de um arquivo.
        
        Args:
            filename: Nome do arquivo
        """
        if not os.path.exists(filename):
            print(f"[AgenteRL] Arquivo {filename} não encontrado.")
            return
        
        try:
            with open(filename, 'rb') as f:
                dados = pickle.load(f)
            
            self.q_tabela = dados.get('q_tabela', {})
            self.epsilon = dados.get('epsilon', 0.1)
            self.melhor_recompensa = dados.get('melhor_recompensa', -float('inf'))
            self.melhor_episodio = dados.get('melhor_episodio', 0)
            self.episodios_treinados = dados.get('episodios_treinados', 0)
            self.exploracoes_total = dados.get('exploracoes_total', 0)
            
            # Atualizar parâmetros se fornecidos
            parametros = dados.get('parametros', {})
            if parametros:
                self.alfa = parametros.get('alfa', self.alfa)
                self.gama = parametros.get('gama', self.gama)
                self.epsilon_decay = parametros.get('epsilon_decay', self.epsilon_decay)
                self.epsilon_min = parametros.get('epsilon_min', self.epsilon_min)
            
            print(f"[AgenteRL] Agente carregado de {filename}")
            print(f"  Estados aprendidos: {len(self.q_tabela)}")
            
        except Exception as e:
            print(f"[AgenteRL] Erro ao carregar agente: {e}")


class AgenteRLMelhorado(AgenteRL):
    """Agente RL com melhorias: Experience Replay e Double Q-Learning."""
    
    def __init__(self, id_agente: str, posicao_inicial: Tuple[int, int], **kwargs):
        """
        Inicializa o agente RL melhorado.
        
        Args:
            id_agente: Identificador do agente
            posicao_inicial: Posição inicial
            **kwargs: Parâmetros adicionais para AgenteRL
        """
        super().__init__(id_agente, posicao_inicial, **kwargs)
        self.tipo = "rl_melhorado"
        
        # Experience Replay
        self.memoria_experiencia = deque(maxlen=2000)
        self.tamanho_lote = 32
        
        # Double Q-Learning
        self.q_tabela_b = {}
        
        # Para rastreamento
        self.contador_aprendizagem = 0
    
    def age(self, modo: str = 'teste') -> str:
        """
        Escolhe ação com Double Q-Learning.
        
        Args:
            modo: Modo de operação
            
        Returns:
            Ação escolhida
        """
        if self.percepcao is None:
            return 'PARAR'
        
        if self.percepcao.get('TERMINOU', False):
            self.terminado = True
            return 'PARAR'
        
        estado = self._extrair_estado()
        
        # Inicializar ambas Q-tables se necessário
        if estado not in self.q_tabela:
            self.q_tabela[estado] = {accao: 0.0 for accao in self.accoes}
        if estado not in self.q_tabela_b:
            self.q_tabela_b[estado] = {accao: 0.0 for accao in self.accoes}
        
        # Escolher ação (ε-greedy com média das duas Q-tables)
        if modo == 'aprendizagem' and random.random() < self.epsilon:
            # Exploração
            accoes_validas = [a for a in self.accoes if a != 'PARAR']
            accao = random.choice(accoes_validas)
            self.exploracoes += 1
            self.exploracoes_total += 1
        else:
            # Exploração: média das duas Q-tables
            q_valores_a = self.q_tabela[estado]
            q_valores_b = self.q_tabela_b[estado]
            
            # Calcular média
            q_valores_media = {}
            for accao in self.accoes:
                if accao != 'PARAR':
                    q_valores_media[accao] = (q_valores_a[accao] + q_valores_b[accao]) / 2
            
            if q_valores_media:
                max_valor = max(q_valores_media.values())
                melhores_accoes = [a for a, v in q_valores_media.items() if v == max_valor]
                accao = random.choice(melhores_accoes)
            else:
                accao = random.choice([a for a in self.accoes if a != 'PARAR'])
        
        # Guardar para aprendizagem
        if modo == 'aprendizagem':
            self.estado_anterior = estado
            self.accao_anterior = accao
        
        return accao
    
    def avaliacaoEstadoAtual(self, recompensa: float) -> None:
        """
        Atualiza usando Experience Replay e Double Q-Learning.
        
        Args:
            recompensa: Recompensa recebida
        """
        self.recompensa_episodio += recompensa
        self.passos_episodio += 1
        
        if self.estado_anterior is None or self.accao_anterior is None:
            return
        
        estado_atual = self._extrair_estado()
        
        # Garantir que estados estão nas Q-tables
        for estado in [self.estado_anterior, estado_atual]:
            if estado not in self.q_tabela:
                self.q_tabela[estado] = {accao: 0.0 for accao in self.accoes}
            if estado not in self.q_tabela_b:
                self.q_tabela_b[estado] = {accao: 0.0 for accao in self.accoes}
        
        # Adicionar experiência à memória
        experiencia = (
            self.estado_anterior,
            self.accao_anterior,
            recompensa,
            estado_atual,
            self.terminado
        )
        self.memoria_experiencia.append(experiencia)
        
        # Q-Learning básico (para comparação)
        q_antigo = self.q_tabela[self.estado_anterior][self.accao_anterior]
        max_q_futuro = max(self.q_tabela[estado_atual].values())
        novo_q = q_antigo + self.alfa * (recompensa + self.gama * max_q_futuro - q_antigo)
        self.q_tabela[self.estado_anterior][self.accao_anterior] = novo_q
        
        # Double Q-Learning update (alternando entre as duas tabelas)
        if random.random() < 0.5:
            # Usar Q_a para selecionar, Q_b para avaliar
            melhor_accao_futuro = max(self.q_tabela[estado_atual].items(), key=lambda x: x[1])[0]
            valor_futuro = self.q_tabela_b[estado_atual][melhor_accao_futuro]
            
            q_antigo_b = self.q_tabela_b[self.estado_anterior][self.accao_anterior]
            novo_q_b = q_antigo_b + self.alfa * (recompensa + self.gama * valor_futuro - q_antigo_b)
            self.q_tabela_b[self.estado_anterior][self.accao_anterior] = novo_q_b
        else:
            # Usar Q_b para selecionar, Q_a para avaliar
            melhor_accao_futuro = max(self.q_tabela_b[estado_atual].items(), key=lambda x: x[1])[0]
            valor_futuro = self.q_tabela[estado_atual][melhor_accao_futuro]
            
            q_antigo = self.q_tabela[self.estado_anterior][self.accao_anterior]
            novo_q = q_antigo + self.alfa * (recompensa + self.gama * valor_futuro - q_antigo)
            self.q_tabela[self.estado_anterior][self.accao_anterior] = novo_q
        
        # Experience Replay (treinar com batch)
        if len(self.memoria_experiencia) >= self.tamanho_lote:
            self._treinar_com_replay()
        
        # Resetar para próximo passo
        self.estado_anterior = None
        self.accao_anterior = None
        self.contador_aprendizagem += 1
    
    def _treinar_com_replay(self) -> None:
        """
        Treina usando um batch da memória de experiência.
        """
        if len(self.memoria_experiencia) < self.tamanho_lote:
            return
        
        # Amostrar batch
        batch = random.sample(self.memoria_experiencia, min(self.tamanho_lote, len(self.memoria_experiencia)))
        
        for estado, accao, recompensa, estado_prox, terminado in batch:
            # Garantir que estados estão nas tabelas
            for e in [estado, estado_prox]:
                if e not in self.q_tabela:
                    self.q_tabela[e] = {a: 0.0 for a in self.accoes}
                if e not in self.q_tabela_b:
                    self.q_tabela_b[e] = {a: 0.0 for a in self.accoes}
            
            # Double Q-Learning update para este exemplo
            if random.random() < 0.5:
                # Selecionar com Q_a, avaliar com Q_b
                if terminado:
                    valor_alvo = recompensa
                else:
                    melhor_accao_prox = max(self.q_tabela[estado_prox].items(), key=lambda x: x[1])[0]
                    valor_alvo = recompensa + self.gama * self.q_tabela_b[estado_prox][melhor_accao_prox]
                
                q_atual = self.q_tabela[estado][accao]
                novo_q = q_atual + self.alfa * (valor_alvo - q_atual)
                self.q_tabela[estado][accao] = novo_q
            else:
                # Selecionar com Q_b, avaliar com Q_a
                if terminado:
                    valor_alvo = recompensa
                else:
                    melhor_accao_prox = max(self.q_tabela_b[estado_prox].items(), key=lambda x: x[1])[0]
                    valor_alvo = recompensa + self.gama * self.q_tabela[estado_prox][melhor_accao_prox]
                
                q_atual = self.q_tabela_b[estado][accao]
                novo_q = q_atual + self.alfa * (valor_alvo - q_atual)
                self.q_tabela_b[estado][accao] = novo_q
    
    def salvar_agente(self, filename: str) -> None:
        """
        Salva o agente melhorado.
        
        Args:
            filename: Nome do arquivo
        """
        dados = {
            'id': self.id,
            'tipo': self.tipo,
            'q_tabela': self.q_tabela,
            'q_tabela_b': self.q_tabela_b,
            'epsilon': self.epsilon,
            'melhor_recompensa': self.melhor_recompensa,
            'melhor_episodio': self.melhor_episodio,
            'episodios_treinados': self.episodios_treinados,
            'exploracoes_total': self.exploracoes_total,
            'memoria_tamanho': len(self.memoria_experiencia),
            'contador_aprendizagem': self.contador_aprendizagem,
            'parametros': {
                'alfa': self.alfa,
                'gama': self.gama,
                'epsilon_decay': self.epsilon_decay,
                'epsilon_min': self.epsilon_min
            },
            'data_criacao': self.data_criacao,
            'ultima_atualizacao': time.time()
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(dados, f)
    
    def carregar_agente(self, filename: str) -> None:
        """
        Carrega o agente melhorado.
        
        Args:
            filename: Nome do arquivo
        """
        if not os.path.exists(filename):
            print(f"[AgenteRLMelhorado] Arquivo {filename} não encontrado.")
            return
        
        with open(filename, 'rb') as f:
            dados = pickle.load(f)
        
        self.q_tabela = dados.get('q_tabela', {})
        self.q_tabela_b = dados.get('q_tabela_b', {})
        self.epsilon = dados.get('epsilon', 0.1)
        self.melhor_recompensa = dados.get('melhor_recompensa', -float('inf'))
        self.melhor_episodio = dados.get('melhor_episodio', 0)
        self.episodios_treinados = dados.get('episodios_treinados', 0)
        self.exploracoes_total = dados.get('exploracoes_total', 0)
        self.contador_aprendizagem = dados.get('contador_aprendizagem', 0)
        
        # Recriar memória de experiência
        self.memoria_experiencia = deque(maxlen=2000)
        
        # Atualizar parâmetros
        parametros = dados.get('parametros', {})
        if parametros:
            self.alfa = parametros.get('alfa', self.alfa)
            self.gama = parametros.get('gama', self.gama)
            self.epsilon_decay = parametros.get('epsilon_decay', self.epsilon_decay)
            self.epsilon_min = parametros.get('epsilon_min', self.epsilon_min)


class GestorAgentes:
    """Gerencia todos os agentes do sistema."""
    
    def __init__(self):
        """Inicializa o gestor de agentes."""
        self.agentes = {}
        self.melhores_agentes = {}
        self.diretorio_agentes = "agentes_salvos"
        
        # Criar diretório se não existir
        os.makedirs(self.diretorio_agentes, exist_ok=True)
    
    def registar_agente(self, agente) -> None:
        """
        Regista um agente no gestor.
        
        Args:
            agente: Agente a registar
        """
        self.agentes[agente.id] = agente
        print(f"[GestorAgentes] Agente {agente.id} registado.")
    
    def salvar_melhor(self, id_agente: str, tipo_ambiente: str) -> None:
        """
        Salva o melhor agente para um tipo de ambiente.
        
        Args:
            id_agente: Identificador do agente
            tipo_ambiente: Tipo de ambiente ('farol' ou 'labirinto')
        """
        if id_agente not in self.agentes:
            print(f"[GestorAgentes] Agente {id_agente} não encontrado.")
            return
        
        agente = self.agentes[id_agente]
        
        # Verificar se é um agente RL
        if not isinstance(agente, (AgenteRL, AgenteRLMelhorado)):
            print(f"[GestorAgentes] Agente {id_agente} não é RL, não será salvo.")
            return
        
        # Criar nome de arquivo
        filename = os.path.join(self.diretorio_agentes, f"melhor_{tipo_ambiente}.pkl")
        
        # Salvar agente
        if isinstance(agente, (AgenteRL, AgenteRLMelhorado)):
            agente.salvar_agente(filename)
        
        # Atualizar registro de melhores agentes
        self.melhores_agentes[tipo_ambiente] = {
            'id': id_agente,
            'tipo': agente.tipo,
            'melhor_recompensa': agente.melhor_recompensa,
            'melhor_episodio': agente.melhor_episodio,
            'filename': filename,
            'timestamp': time.time()
        }
        
        print(f"[GestorAgentes] Melhor agente para {tipo_ambiente} salvo em {filename}")
    
    def carregar_melhor(self, tipo_ambiente: str, id_agente: str = None):
        """
        Carrega o melhor agente para um tipo de ambiente.
        
        Args:
            tipo_ambiente: Tipo de ambiente
            id_agente: Identificador do agente (opcional)
            
        Returns:
            Agente carregado ou None se não existir
        """
        filename = os.path.join(self.diretorio_agentes, f"melhor_{tipo_ambiente}.pkl")
        
        if not os.path.exists(filename):
            print(f"[GestorAgentes] Nenhum agente salvo para {tipo_ambiente}")
            return None
        
        try:
            # Tentar determinar o tipo de agente pelo arquivo
            with open(filename, 'rb') as f:
                dados = pickle.load(f)
            
            tipo_agente = dados.get('tipo', 'rl')
            
            # Criar agente do tipo correto
            if tipo_agente == 'rl_melhorado':
                agente = AgenteRLMelhorado(id_agente or f"Melhor_{tipo_ambiente.capitalize()}", (0, 0))
            else:
                agente = AgenteRL(id_agente or f"Melhor_{tipo_ambiente.capitalize()}", (0, 0))
            
            # Carregar dados
            agente.carregar_agente(filename)
            
            print(f"[GestorAgentes] Agente carregado de {filename}")
            print(f"  ID: {agente.id}")
            print(f"  Tipo: {agente.tipo}")
            print(f"  Estados aprendidos: {len(agente.q_tabela)}")
            print(f"  Melhor recompensa: {agente.melhor_recompensa:.1f}")
            print(f"  Episódios treinados: {agente.episodios_treinados}")
            
            return agente
            
        except Exception as e:
            print(f"[GestorAgentes] Erro ao carregar agente: {e}")
            return None
    
    def listar_agentes_salvos(self) -> List[str]:
        """
        Lista todos os agentes salvos.
        
        Returns:
            Lista com informações dos agentes salvos
        """
        agentes = []
        
        if not os.path.exists(self.diretorio_agentes):
            return agentes
        
        for arquivo in os.listdir(self.diretorio_agentes):
            if arquivo.endswith('.pkl'):
                try:
                    caminho = os.path.join(self.diretorio_agentes, arquivo)
                    with open(caminho, 'rb') as f:
                        dados = pickle.load(f)
                    
                    info = f"{arquivo}: {dados.get('id', 'Desconhecido')} "
                    info += f"(Tipo: {dados.get('tipo', '?')}, "
                    info += f"Recompensa: {dados.get('melhor_recompensa', 0):.1f})"
                    agentes.append(info)
                except:
                    agentes.append(f"{arquivo}: Erro ao carregar")
        
        return agentes
    
    def obter_estatisticas(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do gestor.
        
        Returns:
            Dicionário com estatísticas
        """
        agentes_rl = sum(1 for a in self.agentes.values() if isinstance(a, (AgenteRL, AgenteRLMelhorado)))
        agentes_fixo = sum(1 for a in self.agentes.values() if isinstance(a, AgenteFixo))
        
        return {
            'total_agentes': len(self.agentes),
            'agentes_rl': agentes_rl,
            'agentes_fixo': agentes_fixo,
            'melhores_agentes': self.melhores_agentes.copy()
        }


# Adicionar esta linha para garantir que as classes estão disponíveis
__all__ = ['AgenteBase', 'AgenteFixo', 'AgenteRL', 'AgenteRLMelhorado', 'GestorAgentes']