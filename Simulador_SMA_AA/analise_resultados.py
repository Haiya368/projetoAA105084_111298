#!/usr/bin/env python3
"""
Módulo de análise e visualização de resultados do Simulador SMA.
Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)
Data: Dezembro 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import json
from typing import Dict, List, Any, Optional
import time

# Definir valores padrão se config.py não existir
try:
    from config import CORES, EMOJIS
except ImportError:
    CORES = {'reset': '', 'vermelho': '', 'verde': '', 'azul': '', 'amarelo': '', 'negrito': ''}
    EMOJIS = {'robo': '🤖', 'grafico': '📊', 'sucesso': '✅', 'erro': '❌', 'cerebro': '🧠',
              'alerta': '⚠️', 'salvar': '💾', 'estatisticas': '📈', 'agente': '👤'}


class AnalisadorResultados:
    """Classe para análise e visualização de resultados do treino."""
    
    def __init__(self, diretorio_resultados: str = 'resultados'):
        """
        Inicializa o analisador de resultados.
        
        Args:
            diretorio_resultados: Diretório onde os resultados estão salvos
        """
        self.diretorio_resultados = diretorio_resultados
        os.makedirs(diretorio_resultados, exist_ok=True)
        
        # Configuração de estilo para gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Paleta de cores para gráficos
        self.paleta_cores = {
            'farol': '#FF6B6B',     # Vermelho
            'labirinto': '#4ECDC4',  # Turquesa
            'sucesso': '#1DD1A1',    # Verde
            'falha': '#FF9F43',      # Laranja
            'exploração': '#54A0FF', # Azul
            'explotação': '#5F27CD'  # Roxo
        }
    
    def carregar_dados(self, arquivo_csv: str) -> Optional[pd.DataFrame]:
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            arquivo_csv: Caminho para o arquivo CSV
            
        Returns:
            DataFrame com os dados ou None se erro
        """
        try:
            if not os.path.exists(arquivo_csv):
                print(f"{EMOJIS['erro']} Arquivo não encontrado: {arquivo_csv}")
                return None
            
            df = pd.read_csv(arquivo_csv)
            print(f"{EMOJIS['sucesso']} Dados carregados: {len(df)} registos")
            return df
        except Exception as e:
            print(f"{EMOJIS['erro']} Erro ao carregar dados: {e}")
            return None
    
    def plotar_curva_aprendizagem(self, df: pd.DataFrame, 
                                 tipo_ambiente: str,
                                 salvar: bool = True,
                                 mostrar: bool = True):
        """
        Plota a curva de aprendizagem.
        
        Args:
            df: DataFrame com os dados
            tipo_ambiente: Tipo de ambiente ('farol' ou 'labirinto')
            salvar: Se True, salva o gráfico
            mostrar: Se True, mostra o gráfico
        """
        if df is None or df.empty:
            print(f"{EMOJIS['alerta']} Nenhum dado para plotar curva de aprendizagem")
            return
        
        # Agrupar por episódio
        if 'episodio' not in df.columns:
            print(f"{EMOJIS['alerta']} Coluna 'episodio' não encontrada")
            return
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Curva de Aprendizagem - {tipo_ambiente.upper()}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Recompensa por episódio
        ax1 = axes[0, 0]
        recompensa_por_episodio = df.groupby('episodio')['recompensa'].mean()
        recompensa_suavizada = recompensa_por_episodio.rolling(window=10, min_periods=1).mean()
        
        ax1.plot(recompensa_por_episodio.index, recompensa_por_episodio.values, 
                alpha=0.3, color=self.paleta_cores[tipo_ambiente], label='Original')
        ax1.plot(recompensa_suavizada.index, recompensa_suavizada.values, 
                linewidth=2, color=self.paleta_cores[tipo_ambiente], label='Média móvel (10)')
        
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa Total')
        ax1.set_title('Recompensa por Episódio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Taxa de sucesso
        ax2 = axes[0, 1]
        if 'sucesso' in df.columns:
            taxa_sucesso = df.groupby('episodio')['sucesso'].mean() * 100
            taxa_sucesso_suavizada = taxa_sucesso.rolling(window=10, min_periods=1).mean()
            
            ax2.plot(taxa_sucesso.index, taxa_sucesso.values, alpha=0.3, 
                    color=self.paleta_cores['sucesso'], label='Original')
            ax2.plot(taxa_sucesso_suavizada.index, taxa_sucesso_suavizada.values, 
                    linewidth=2, color=self.paleta_cores['sucesso'], label='Média móvel (10)')
            
            ax2.set_xlabel('Episódio')
            ax2.set_ylabel('Taxa de Sucesso (%)')
            ax2.set_title('Taxa de Sucesso por Episódio')
            ax2.set_ylim([0, 105])
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Passos por episódio
        ax3 = axes[1, 0]
        if 'passos' in df.columns:
            passos_por_episodio = df.groupby('episodio')['passos'].mean()
            passos_suavizados = passos_por_episodio.rolling(window=10, min_periods=1).mean()
            
            ax3.plot(passos_por_episodio.index, passos_por_episodio.values, alpha=0.3,
                    color=self.paleta_cores['exploração'], label='Original')
            ax3.plot(passos_suavizados.index, passos_suavizados.values, linewidth=2,
                    color=self.paleta_cores['exploração'], label='Média móvel (10)')
            
            ax3.set_xlabel('Episódio')
            ax3.set_ylabel('Número de Passos')
            ax3.set_title('Eficiência (Passos por Episódio)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Colisões por episódio
        ax4 = axes[1, 1]
        if 'colisoes' in df.columns:
            colisoes_por_episodio = df.groupby('episodio')['colisoes'].mean()
            colisoes_suavizadas = colisoes_por_episodio.rolling(window=10, min_periods=1).mean()
            
            ax4.plot(colisoes_por_episodio.index, colisoes_por_episodio.values, alpha=0.3,
                    color=self.paleta_cores['falha'], label='Original')
            ax4.plot(colisoes_suavizadas.index, colisoes_suavizadas.values, linewidth=2,
                    color=self.paleta_cores['falha'], label='Média móvel (10)')
            
            ax4.set_xlabel('Episódio')
            ax4.set_ylabel('Número de Colisões')
            ax4.set_title('Segurança (Colisões por Episódio)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # Se não houver colisões, mostrar epsilon
            if 'epsilon' in df.columns:
                epsilon_por_episodio = df.groupby('episodio')['epsilon'].mean()
                ax4.plot(epsilon_por_episodio.index, epsilon_por_episodio.values,
                        linewidth=2, color=self.paleta_cores['explotação'])
                ax4.set_xlabel('Episódio')
                ax4.set_ylabel('Valor do Epsilon')
                ax4.set_title('Exploração vs Exploração')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar gráfico
        if salvar:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            nome_arquivo = os.path.join(self.diretorio_resultados, 
                                       f'curva_aprendizagem_{tipo_ambiente}_{timestamp}.png')
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"{EMOJIS['salvar']} Gráfico salvo em: {nome_arquivo}")
        
        # Mostrar gráfico
        if mostrar:
            plt.show()
        else:
            plt.close()

    def plotar_comparacao_agentes(self, dfs: Dict[str, pd.DataFrame], 
                                 salvar: bool = True, mostrar: bool = True):
        """
        Plota comparação entre múltiplos agentes.
        
        Args:
            dfs: Dicionário com DataFrames para cada agente
            salvar: Se True, salva o gráfico
            mostrar: Se True, mostra o gráfico
        """
        if not dfs:
            print(f"{EMOJIS['alerta']} Nenhum dado para plotar comparação")
            return
        
        # Criar figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Comparação de Agentes', fontsize=16, fontweight='bold')
        
        # Cores para diferentes agentes
        cores = plt.cm.tab10(np.linspace(0, 1, len(dfs)))
        
        # 1. Recompensa média por episódio
        ax1 = axes[0, 0]
        for i, (nome_agente, df) in enumerate(dfs.items()):
            if 'recompensa' in df.columns and 'episodio' in df.columns:
                recompensa_media = df.groupby('episodio')['recompensa'].mean()
                recompensa_suavizada = recompensa_media.rolling(window=10, min_periods=1).mean()
                ax1.plot(recompensa_suavizada.index, recompensa_suavizada.values, 
                        label=nome_agente, color=cores[i], linewidth=2)
        
        ax1.set_xlabel('Episódio')
        ax1.set_ylabel('Recompensa Média')
        ax1.set_title('Recompensa por Episódio (Média móvel 10)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Taxa de sucesso acumulada
        ax2 = axes[0, 1]
        for i, (nome_agente, df) in enumerate(dfs.items()):
            if 'sucesso' in df.columns and 'episodio' in df.columns:
                # Calcular taxa de sucesso acumulada
                df_agrupado = df.groupby('episodio')['sucesso'].mean()
                taxa_acumulada = df_agrupado.expanding().mean() * 100
                ax2.plot(taxa_acumulada.index, taxa_acumulada.values, 
                        label=nome_agente, color=cores[i], linewidth=2)
        
        ax2.set_xlabel('Episódio')
        ax2.set_ylabel('Taxa de Sucesso Acumulada (%)')
        ax2.set_title('Taxa de Sucesso Acumulada')
        ax2.set_ylim([0, 105])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Eficiência (passos por episódio)
        ax3 = axes[1, 0]
        for i, (nome_agente, df) in enumerate(dfs.items()):
            if 'passos' in df.columns and 'episodio' in df.columns:
                passos_media = df.groupby('episodio')['passos'].mean()
                passos_suavizada = passos_media.rolling(window=10, min_periods=1).mean()
                ax3.plot(passos_suavizada.index, passos_suavizada.values, 
                        label=nome_agente, color=cores[i], linewidth=2)
        
        ax3.set_xlabel('Episódio')
        ax3.set_ylabel('Passos por Episódio')
        ax3.set_title('Eficiência (Passos por Episódio)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Box plot de recompensas finais
        ax4 = axes[1, 1]
        recompensas_finais = []
        nomes_agentes = []
        
        for nome_agente, df in dfs.items():
            if 'recompensa' in df.columns:
                # Pegar últimos 20% dos episódios
                total_episodios = df['episodio'].max() if 'episodio' in df.columns else len(df)
                episodios_finais = int(total_episodios * 0.2)
                if episodios_finais < 5:
                    episodios_finais = min(5, total_episodios)
                
                if 'episodio' in df.columns:
                    df_final = df[df['episodio'] >= (total_episodios - episodios_finais)]
                else:
                    df_final = df.tail(episodios_finais)
                
                if not df_final.empty:
                    recompensas_finais.append(df_final['recompensa'].values)
                    nomes_agentes.append(nome_agente)
        
        if recompensas_finais:
            box = ax4.boxplot(recompensas_finais, labels=nomes_agentes, patch_artist=True)
            
            # Colorir os boxes
            for patch, color in zip(box['boxes'], cores[:len(recompensas_finais)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax4.set_ylabel('Recompensa')
            ax4.set_title('Distribuição de Recompensas (Últimos 20% dos episódios)')
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, 'Sem dados suficientes', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distribuição de Recompensas')
        
        plt.tight_layout()
        
        # Salvar gráfico
        if salvar:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            nome_arquivo = os.path.join(self.diretorio_resultados, 
                                       f'comparacao_agentes_{timestamp}.png')
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"{EMOJIS['salvar']} Gráfico de comparação salvo em: {nome_arquivo}")
        
        # Mostrar gráfico
        if mostrar:
            plt.show()
        else:
            plt.close()
    
    def gerar_relatorio_html(self, dfs: Dict[str, pd.DataFrame], 
                           nome_relatorio: str = "relatorio_sma"):
        """
        Gera um relatório HTML com os resultados.
        
        Args:
            dfs: Dicionário com DataFrames para cada experiência
            nome_relatorio: Nome base para o relatório
        """
        # Verificar se temos dados
        if not dfs:
            print(f"{EMOJIS['alerta']} Nenhum dado para gerar relatório")
            return None
        
        # Template HTML simples (sem Jinja2 para não exigir instalação adicional)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        html_parts = []
        html_parts.append('''<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório SMA - ''' + timestamp + '''</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #333; border-bottom: 2px solid #4ECDC4; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .card { background: #f9f9f9; border: 1px solid #ddd; border-radius: 5px; padding: 20px; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4ECDC4; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .success { color: #1DD1A1; font-weight: bold; }
        .warning { color: #FF9F43; }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 24px; font-weight: bold; color: #333; }
        .metric-label { font-size: 14px; color: #666; }
        .img-container { text-align: center; margin: 30px 0; }
        img { max-width: 90%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>📊 Relatório do Simulador SMA</h1>
    <p><strong>Data:</strong> ''' + timestamp + '''</p>
    <p><strong>Total de experiências:</strong> ''' + str(len(dfs)) + '''</p>''')
        
        for exp_nome, df in dfs.items():
            # Calcular métricas básicas
            metrics = {}
            
            if 'recompensa' in df.columns:
                metrics['recompensa_media'] = {
                    'formatted': f"{df['recompensa'].mean():.1f}",
                    'label': 'Recompensa Média'
                }
            
            if 'sucesso' in df.columns:
                taxa_sucesso = df['sucesso'].mean() * 100
                metrics['taxa_sucesso'] = {
                    'formatted': f"{taxa_sucesso:.1f}%",
                    'label': 'Taxa de Sucesso'
                }
            
            if 'passos' in df.columns:
                metrics['passos_medios'] = {
                    'formatted': f"{df['passos'].mean():.1f}",
                    'label': 'Passos Médios'
                }
            
            if 'epsilon' in df.columns:
                metrics['epsilon_final'] = {
                    'formatted': f"{df['epsilon'].iloc[-1] if len(df) > 0 else 0:.3f}",
                    'label': 'Epsilon Final'
                }
            
            # Começar card da experiência
            html_parts.append(f'''
    <div class="card">
        <h2>🎯 Experiência: {exp_nome}</h2>
        
        <h3>📈 Métricas Principais</h3>
        <div>''')
            
            for key, value in metrics.items():
                html_parts.append(f'''
            <div class="metric">
                <div class="metric-value">{value['formatted']}</div>
                <div class="metric-label">{value['label']}</div>
            </div>''')
            
            html_parts.append('''
        </div>
        
        <h3>📋 Estatísticas Detalhadas</h3>
        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
                <th>Descrição</th>
            </tr>''')
            
            # Adicionar estatísticas
            stats = []
            for col in df.columns:
                if col not in ['episodio', 'agente', 'tipo', 'modo']:
                    stats.append({
                        'name': col.replace('_', ' ').title(),
                        'value': f"{df[col].mean():.2f} ± {df[col].std():.2f}",
                        'description': f"Média e desvio padrão de {col}"
                    })
            
            for stat in stats[:10]:  # Limitar a 10 estatísticas
                html_parts.append(f'''
            <tr>
                <td>{stat['name']}</td>
                <td>{stat['value']}</td>
                <td>{stat['description']}</td>
            </tr>''')
            
            html_parts.append('''
        </table>
        
        <h3>📊 Informações do Dataset</h3>
        <table>
            <tr>
                <td>Total de registos</td>
                <td>''' + str(len(df)) + '''</td>
            </tr>
            <tr>
                <td>Colunas disponíveis</td>
                <td>''' + ', '.join(df.columns.tolist()) + '''</td>
            </tr>
        </table>
    </div>''')
        
        # Fechar HTML
        html_parts.append('''
    <hr>
    <footer>
        <p>Simulador SMA - Agentes Autónomos - ISCTE</p>
        <p>Autores: Emanuel Fernandes (105084), Andreia Fonseca (111298)</p>
    </footer>
</body>
</html>''')
        
        # Juntar todas as partes
        html_content = ''.join(html_parts)
        
        # Salvar HTML
        nome_arquivo = os.path.join(self.diretorio_resultados, f"{nome_relatorio}.html")
        with open(nome_arquivo, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"{EMOJIS['salvar']} Relatório HTML gerado em: {nome_arquivo}")
        return nome_arquivo
    
    def analisar_arquivo_resultados(self, arquivo_csv: str):
        """
        Analisa um arquivo de resultados e gera visualizações.
        
        Args:
            arquivo_csv: Caminho para o arquivo CSV
        """
        print(f"\n{EMOJIS['estatisticas']} ANALISANDO RESULTADOS: {arquivo_csv}")
        print("="*60)
        
        # Carregar dados
        df = self.carregar_dados(arquivo_csv)
        if df is None:
            return
        
        # Informações básicas
        print(f"Total de registos: {len(df)}")
        if 'episodio' in df.columns:
            print(f"Total de episódios: {df['episodio'].nunique()}")
        if 'agente' in df.columns:
            print(f"Agentes únicos: {', '.join(df['agente'].unique())}")
        
        # Determinar tipo de ambiente pelo nome do arquivo
        arquivo_lower = arquivo_csv.lower()
        if 'labirinto' in arquivo_lower:
            tipo_ambiente = 'labirinto'
        elif 'farol' in arquivo_lower:
            tipo_ambiente = 'farol'
        else:
            # Tentar inferir pelo conteúdo
            if 'distancia_saida' in df.columns:
                tipo_ambiente = 'labirinto'
            elif 'distancia_farol' in df.columns:
                tipo_ambiente = 'farol'
            else:
                tipo_ambiente = 'desconhecido'
        
        if tipo_ambiente != 'desconhecido':
            # Gerar curva de aprendizagem
            self.plotar_curva_aprendizagem(df, tipo_ambiente, salvar=True, mostrar=False)
            
            # Gerar estatísticas resumo
            self.gerar_estatisticas_resumo(df, arquivo_csv)
            
            # Gerar relatório se houver dados suficientes
            if len(df) > 10:
                nome_base = os.path.basename(arquivo_csv).replace('.csv', '')
                self.gerar_relatorio_html({nome_base: df}, 
                                         f"relatorio_{nome_base}")
        else:
            print(f"{EMOJIS['alerta']} Não foi possível determinar o tipo de ambiente")
        
        print(f"{EMOJIS['sucesso']} Análise concluída!")
    
    def gerar_estatisticas_resumo(self, df: pd.DataFrame, arquivo_csv: str):
        """
        Gera estatísticas resumo dos resultados.
        
        Args:
            df: DataFrame com os dados
            arquivo_csv: Nome do arquivo original
        """
        if df is None or df.empty:
            return
        
        print(f"\n{EMOJIS['estatisticas']} ESTATÍSTICAS RESUMO")
        print("-"*60)
        
        # Estatísticas gerais
        if 'recompensa' in df.columns:
            print(f"Recompensa:")
            print(f"  Média: {df['recompensa'].mean():.2f}")
            print(f"  Mediana: {df['recompensa'].median():.2f}")
            print(f"  Máximo: {df['recompensa'].max():.2f}")
            print(f"  Mínimo: {df['recompensa'].min():.2f}")
            print(f"  Desvio Padrão: {df['recompensa'].std():.2f}")
        
        if 'sucesso' in df.columns:
            taxa_sucesso = df['sucesso'].mean() * 100
            print(f"\nTaxa de Sucesso: {taxa_sucesso:.2f}%")
            print(f"  Episódios com sucesso: {int(df['sucesso'].sum())}")
            print(f"  Total de episódios: {len(df)}")
        
        if 'passos' in df.columns:
            print(f"\nEficiência:")
            print(f"  Passos médios por episódio: {df['passos'].mean():.2f}")
        
        if 'colisoes' in df.columns:
            print(f"\nSegurança:")
            print(f"  Colisões médias por episódio: {df['colisoes'].mean():.2f}")
        
        if 'epsilon' in df.columns:
            print(f"\nExploração:")
            print(f"  Epsilon inicial: {df['epsilon'].iloc[0] if len(df) > 0 else 'N/A':.3f}")
            print(f"  Epsilon final: {df['epsilon'].iloc[-1] if len(df) > 0 else 'N/A':.3f}")
        
        # Estatísticas por agente se houver múltiplos agentes
        if 'agente' in df.columns and df['agente'].nunique() > 1:
            print(f"\n{EMOJIS['agente']} COMPARAÇÃO POR AGENTE")
            print("-"*60)
            
            for agente in df['agente'].unique():
                df_agente = df[df['agente'] == agente]
                print(f"\nAgente: {agente}")
                print(f"  Episódios: {len(df_agente)}")
                
                if 'recompensa' in df_agente.columns:
                    print(f"  Recompensa média: {df_agente['recompensa'].mean():.2f}")
                
                if 'sucesso' in df_agente.columns:
                    taxa = df_agente['sucesso'].mean() * 100
                    print(f"  Taxa de sucesso: {taxa:.2f}%")
        
        # Salvar estatísticas em arquivo
        self.salvar_estatisticas(df, arquivo_csv)
    
    def salvar_estatisticas(self, df: pd.DataFrame, arquivo_csv: str):
        """
        Salva estatísticas em arquivo JSON.
        
        Args:
            df: DataFrame com os dados
            arquivo_csv: Nome do arquivo original
        """
        estatisticas = {}
        
        # Coletar estatísticas
        if 'recompensa' in df.columns:
            estatisticas['recompensa'] = {
                'media': float(df['recompensa'].mean()),
                'mediana': float(df['recompensa'].median()),
                'maximo': float(df['recompensa'].max()),
                'minimo': float(df['recompensa'].min()),
                'desvio_padrao': float(df['recompensa'].std())
            }
        
        if 'sucesso' in df.columns:
            estatisticas['sucesso'] = {
                'taxa': float(df['sucesso'].mean() * 100),
                'total_sucessos': int(df['sucesso'].sum()),
                'total_episodios': len(df)
            }
        
        if 'passos' in df.columns:
            estatisticas['passos'] = {
                'media': float(df['passos'].mean()),
                'total': int(df['passos'].sum())
            }
        
        if 'colisoes' in df.columns:
            estatisticas['colisoes'] = {
                'media': float(df['colisoes'].mean()),
                'total': int(df['colisoes'].sum())
            }
        
        if 'epsilon' in df.columns:
            estatisticas['epsilon'] = {
                'inicial': float(df['epsilon'].iloc[0]) if len(df) > 0 else 0.0,
                'final': float(df['epsilon'].iloc[-1]) if len(df) > 0 else 0.0
            }
        
        # Salvar em JSON
        nome_base = os.path.basename(arquivo_csv).replace('.csv', '')
        nome_json = os.path.join(self.diretorio_resultados, f"estatisticas_{nome_base}.json")
        
        with open(nome_json, 'w', encoding='utf-8') as f:
            json.dump(estatisticas, f, indent=4, ensure_ascii=False)
        
        print(f"{EMOJIS['salvar']} Estatísticas salvas em: {nome_json}")


def menu_analise_resultados():
    """Menu para análise de resultados."""
    
    analisador = AnalisadorResultados()
    
    while True:
        print(f"\n{'='*70}")
        print(f"📊 MENU DE ANÁLISE DE RESULTADOS")
        print(f"{'='*70}")
        print("1. Analisar arquivo de resultados específico")
        print("2. Analisar todos os arquivos de resultados")
        print("3. Comparar múltiplos agentes/experiências")
        print("4. Gerar relatório HTML")
        print("5. Listar arquivos de resultados disponíveis")
        print("0. Voltar ao menu principal")
        
        escolha = input(f"\nEscolha uma opção (0-5): ").strip()
        
        if escolha == '0':
            break
        
        elif escolha == '1':
            arquivos = glob.glob("resultados_*.csv") + glob.glob("aprendizagem_*.csv") + glob.glob("resultados/*.csv")
            if not arquivos:
                print(f"⚠️  Nenhum arquivo de resultados encontrado.")
                print("   Execute primeiro um treino para gerar dados.")
                continue
            
            print(f"\nArquivos disponíveis:")
            for i, arquivo in enumerate(arquivos, 1):
                tamanho = os.path.getsize(arquivo) / 1024  # KB
                print(f"  {i}. {arquivo} ({tamanho:.1f} KB)")
            
            try:
                idx = int(input(f"\nSelecione o número do arquivo (1-{len(arquivos)}): ")) - 1
                if 0 <= idx < len(arquivos):
                    analisador.analisar_arquivo_resultados(arquivos[idx])
                    print("\nPressione Enter para continuar...")
                    input()
                else:
                    print(f"❌ Seleção inválida.")
            except ValueError:
                print(f"❌ Entrada inválida.")
        
        elif escolha == '2':
            arquivos = glob.glob("resultados_*.csv") + glob.glob("aprendizagem_*.csv") + glob.glob("resultados/*.csv")
            if not arquivos:
                print(f"⚠️  Nenhum arquivo de resultados encontrado.")
                print("   Execute primeiro um treino para gerar dados.")
                continue
            
            print(f"\nAnalisando {len(arquivos)} arquivos...")
            for arquivo in arquivos:
                analisador.analisar_arquivo_resultados(arquivo)
                print("-" * 60)
            
            print("\nAnálise de todos os arquivos concluída!")
            print("Pressione Enter para continuar...")
            input()
        
        elif escolha == '3':
            arquivos = glob.glob("resultados_*.csv") + glob.glob("aprendizagem_*.csv") + glob.glob("resultados/*.csv")
            if not arquivos or len(arquivos) < 2:
                print(f"⚠️  É necessário pelo menos 2 arquivos para comparação.")
                print("   Execute primeiro treinos para gerar dados.")
                continue
            
            print(f"\nSelecione 2 ou mais arquivos para comparar:")
            for i, arquivo in enumerate(arquivos, 1):
                tamanho = os.path.getsize(arquivo) / 1024
                print(f"  {i}. {arquivo} ({tamanho:.1f} KB)")
            
            selecionados = input("\nDigite os números separados por vírgula (ex: 1,2,3): ").strip()
            
            try:
                indices = [int(x.strip()) - 1 for x in selecionados.split(',')]
                arquivos_selecionados = [arquivos[i] for i in indices if 0 <= i < len(arquivos)]
                
                if len(arquivos_selecionados) >= 2:
                    # Carregar dados
                    dfs = {}
                    for arquivo in arquivos_selecionados:
                        nome = os.path.basename(arquivo).replace('.csv', '')
                        df = analisador.carregar_dados(arquivo)
                        if df is not None:
                            dfs[nome] = df
                    
                    if len(dfs) >= 2:
                        analisador.plotar_comparacao_agentes(dfs, salvar=True, mostrar=True)
                        print("\nPressione Enter para continuar...")
                        input()
                    else:
                        print(f"⚠️  Não foi possível carregar dados suficientes para comparação.")
                else:
                    print(f"❌ É necessário selecionar pelo menos 2 arquivos.")
            except Exception as e:
                print(f"❌ Erro ao processar seleção: {e}")
        
        elif escolha == '4':
            arquivos = glob.glob("resultados_*.csv") + glob.glob("aprendizagem_*.csv") + glob.glob("resultados/*.csv")
            if not arquivos:
                print(f"⚠️  Nenhum arquivo de resultados encontrado.")
                print("   Execute primeiro um treino para gerar dados.")
                continue
            
            # Carregar todos os arquivos (limitado a 5 para não sobrecarregar)
            dfs = {}
            for arquivo in arquivos[:5]:
                nome = os.path.basename(arquivo).replace('.csv', '')
                df = analisador.carregar_dados(arquivo)
                if df is not None:
                    dfs[nome] = df
            
            if dfs:
                analisador.gerar_relatorio_html(dfs)
                print("\nPressione Enter para continuar...")
                input()
            else:
                print(f"⚠️  Não foi possível carregar dados para o relatório.")
        
        elif escolha == '5':
            arquivos = glob.glob("resultados_*.csv") + glob.glob("aprendizagem_*.csv") + glob.glob("resultados/*.csv")
            if not arquivos:
                print(f"⚠️  Nenhum arquivo de resultados encontrado.")
                print("   Execute primeiro um treino para gerar dados.")
                continue
            
            print(f"\nArquivos de resultados disponíveis ({len(arquivos)}):")
            for i, arquivo in enumerate(arquivos, 1):
                tamanho = os.path.getsize(arquivo) / 1024  # KB
                linhas = 0
                try:
                    with open(arquivo, 'r') as f:
                        linhas = sum(1 for line in f) - 1  # Subtrair cabeçalho
                except:
                    linhas = "?"
                
                print(f"  {i}. {arquivo}")
                print(f"      Tamanho: {tamanho:.1f} KB | Linhas: {linhas}")
        
        else:
            print(f"❌ Opção inválida.")


if __name__ == "__main__":
    # Teste do analisador
    menu_analise_resultados()