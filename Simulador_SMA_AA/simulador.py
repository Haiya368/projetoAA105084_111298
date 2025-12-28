#!/usr/bin/env python3
"""
SIMULADOR.PY CORRIGIDO - Versão funcional simplificada
"""

import time
import pandas as pd
from typing import Dict, Any, Tuple
import os

class Simulador:
    """Simulador simplificado para SMA."""
    
    def __init__(self, ambiente):
        self.ambiente = ambiente
        self.agentes = {}
        self.historico = []
    
    def adicionar_agente(self, agente, id_agente: str, posicao: Tuple[int, int]):
        """Adiciona um agente ao simulador."""
        self.agentes[id_agente] = agente
        self.ambiente.posicoes_agentes[id_agente] = posicao
        agente.posicao = posicao
    
    def executar_episodio(self, max_passos: int, modo: str = 'aprendizagem'):
        """Executa um episódio de simulação."""
        
        resultados = {}
        
        for agente_id, agente in self.agentes.items():
            # Resetar agente
            agente.reiniciar()
            
            # Executar passos
            recompensa_total = 0
            passos = 0
            sucesso = False
            
            for passo in range(max_passos):
                # Observar
                obs = self.ambiente.observacaoPara(agente_id)
                if obs is None or obs.get('TERMINOU', False):
                    break
                
                agente.observacao(obs)
                
                # Agir
                acao = agente.age(modo=modo)
                
                # Executar no ambiente
                resultado = self.ambiente.agir(acao, agente_id)
                
                # Aprender (se modo aprendizagem)
                if modo == 'aprendizagem':
                    prox_obs = self.ambiente.observacaoPara(agente_id)
                    agente.avaliacaoEstadoAtual(resultado['recompensa'])
                
                # Atualizar estatísticas
                recompensa_total += resultado['recompensa']
                passos += 1
                
                # Verificar se terminou
                if resultado.get('terminou', False):
                    sucesso = True
                    break
            
            # Guardar resultados
            resultados[agente_id] = {
                'recompensa': recompensa_total,
                'passos': passos,
                'sucesso': sucesso
            }
        
        return resultados
    
    def executar_treinamento(self, episodios: int, max_passos: int):
        """Executa múltiplos episódios de treinamento."""
        
        historico = []
        
        for episodio in range(episodios):
            # Executar episódio
            resultados = self.executar_episodio(max_passos, modo='aprendizagem')
            
            # Atualizar agentes
            for agente_id, agente in self.agentes.items():
                if hasattr(agente, 'atualizar_melhor'):
                    agente.atualizar_melhor(episodio)
            
            # Guardar no histórico
            for agente_id, resultado in resultados.items():
                historico.append({
                    'episodio': episodio,
                    'agente': agente_id,
                    'recompensa': resultado['recompensa'],
                    'passos': resultado['passos'],
                    'sucesso': resultado['sucesso'],
                    'epsilon': self.agentes[agente_id].epsilon if hasattr(self.agentes[agente_id], 'epsilon') else 0
                })
            
            # Progresso
            if (episodio + 1) % 10 == 0:
                taxa_sucesso = sum([h['sucesso'] for h in historico[-10:]]) / 10 * 100
                print(f"Episódio {episodio+1}/{episodios} | Sucesso: {taxa_sucesso:.1f}%")
        
        return pd.DataFrame(historico)