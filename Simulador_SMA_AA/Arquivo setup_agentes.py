#!/usr/bin/env python3
"""
Script para copiar automaticamente agentes treinados.
Execute após o treino automático de 10.000 episódios.
"""

import os
import shutil

def copiar_agentes():
    """Copia os agentes treinados 10K para os nomes padrão."""
    
    print("🔧 CONFIGURAÇÃO DE AGENTES TREINADOS")
    print("="*60)
    
    agentes = [
        ('melhor_labirinto_10K.pkl', 'melhor_labirinto.pkl'),
        ('melhor_farol_10K.pkl', 'melhor_farol.pkl')
    ]
    
    for origem, destino in agentes:
        caminho_origem = os.path.join('agentes_salvos', origem)
        caminho_destino = os.path.join('agentes_salvos', destino)
        
        if os.path.exists(caminho_origem):
            shutil.copy2(caminho_origem, caminho_destino)
            print(f"✅ Copiado: {origem} → {destino}")
        else:
            print(f"⚠️  Não encontrado: {origem}")
    
    print("\n✅ Configuração concluída!")
    print("   Agora pode usar as demonstrações visuais normalmente.")

if __name__ == "__main__":
    copiar_agentes()