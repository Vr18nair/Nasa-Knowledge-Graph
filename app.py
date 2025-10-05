import streamlit as st
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from collections import Counter
import re

st.set_page_config(page_title="NASA Bioscience Knowledge Graph", layout="wide")

st.title("🔬 NASA Bioscience Knowledge Graph Explorer")
st.markdown("**Interactive analysis of relationships extracted from NASA bioscience research papers**")

# Load data
@st.cache_data
def load_data():
    G = nx.read_graphml("/content/nasa_merged_graph.graphml")
    nodes_df = pd.read_csv("/content/nasa_nodes.csv")
    edges_df = pd.read_csv("/content/nasa_edges.csv")
    return G, nodes_df, edges_df

try:
    G, nodes_df, edges_df = load_data()
except FileNotFoundError:
    st.error("⚠️ Data files not found. Please ensure the knowledge graph has been generated.")
    st.stop()

# Sidebar
st.sidebar.header("📊 Graph Statistics")
st.sidebar.metric("Total Entities", len(G.nodes()))
st.sidebar.metric("Total Relationships", len(G.edges()))
st.sidebar.metric("Average Connections", f"{2*len(G.edges())/len(G.nodes()):.1f}")

# Most connected entities
degree_dict = dict(G.degree())
top_entities = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:5]
st.sidebar.markdown("### 🔝 Most Connected Entities")
for entity, degree in top_entities:
    st.sidebar.markdown(f"- **{entity}**: {degree} connections")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview", 
    "🔍 Entity Search", 
    "🔗 Relationships", 
    "📊 Network Analysis",
    "📥 Data Export"
])

# TAB 1: Overview
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entity Types Distribution")
        entity_types = nodes_df['label'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=entity_types.index, 
            values=entity_types.values,
            hole=0.3
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top Relationship Types")
        rel_types = edges_df['predicate'].value_counts().head(10)
        fig = go.Figure(data=[go.Bar(
            x=rel_types.values,
            y=rel_types.index,
            orientation='h'
        )])
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📚 Source Documents")
    all_sources = []
    for sources in edges_df['sources'].dropna():
        all_sources.extend(str(sources).split(','))
    source_counts = Counter(all_sources)
    
    source_df = pd.DataFrame([
        {"Document": doc.strip(), "Relationships": count} 
        for doc, count in source_counts.most_common(10)
    ])
    st.dataframe(source_df, use_container_width=True)

# TAB 2: Entity Search
with tab2:
    st.subheader("🔍 Search for Entities")
    
    search_term = st.text_input(
        "Enter entity name or keyword:",
        placeholder="e.g., protein, gene, cell, arabidopsis"
    )
    
    if search_term:
        matches = nodes_df[
            nodes_df['node'].str.contains(search_term, case=False, na=False)
        ]
        
        st.markdown(f"### Found {len(matches)} matching entities")
        
        if len(matches) > 0:
            for idx, row in matches.head(20).iterrows():
                with st.expander(f"**{row['node']}** ({row['label']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Ontology Mappings:**")
                        st.write(f"- NCBI: {row.get('ncbi', 'Unknown')}")
                        st.write(f"- GO: {row.get('go', 'Unknown')}")
                    
                    with col2:
                        node_name = row['node']
                        related_out = edges_df[edges_df['subject'] == node_name]
                        related_in = edges_df[edges_df['object'] == node_name]
                        
                        st.markdown(f"**Connections: {len(related_out) + len(related_in)}**")
                        st.write(f"- Outgoing: {len(related_out)}")
                        st.write(f"- Incoming: {len(related_in)}")
                    
                    if len(related_out) > 0:
                        st.markdown("**Sample Relationships:**")
                        for _, edge in related_out.head(5).iterrows():
                            st.markdown(f"➡️ `{edge['subject']}` **{edge['predicate']}** `{edge['object']}`")
        else:
            st.info("No matches found. Try a different search term.")

# TAB 3: Relationships
with tab3:
    st.subheader("🔗 Explore Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        relationship_type = st.selectbox(
            "Filter by relationship type:",
            ["All"] + sorted(edges_df['predicate'].unique().tolist())
        )
    
    with col2:
        min_weight = st.slider(
            "Minimum relationship strength:",
            min_value=1,
            max_value=int(edges_df['weight'].max()),
            value=1
        )
    
    filtered_edges = edges_df[edges_df['weight'] >= min_weight].copy()
    if relationship_type != "All":
        filtered_edges = filtered_edges[filtered_edges['predicate'] == relationship_type]
    
    st.markdown(f"### Showing {len(filtered_edges)} relationships")
    
    display_df = filtered_edges[['subject', 'predicate', 'object', 'weight']].head(100)
    display_df.columns = ['Entity 1', 'Relationship', 'Entity 2', 'Strength']
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("### 💎 High-Confidence Relationships")
    st.markdown("*Relationships found across multiple sources*")
    
    consensus = filtered_edges[filtered_edges['weight'] > 2].sort_values('weight', ascending=False)
    if len(consensus) > 0:
        for _, row in consensus.head(10).iterrows():
            st.markdown(
                f"**{row['subject']}** ➡️ *{row['predicate']}* ➡️ **{row['object']}** "
                f"(strength: {row['weight']})"
            )
    else:
        st.info("No high-confidence relationships found with current filters.")

# TAB 4: Network Analysis
with tab4:
    st.subheader("📊 Network Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Network Density", f"{nx.density(G):.4f}")
    
    with col2:
        if nx.is_connected(G.to_undirected()):
            diameter = nx.diameter(G.to_undirected())
            st.metric("Network Diameter", diameter)
        else:
            components = nx.number_connected_components(G.to_undirected())
            st.metric("Connected Components", components)
    
    with col3:
        avg_clustering = nx.average_clustering(G.to_undirected())
        st.metric("Avg Clustering", f"{avg_clustering:.4f}")
    
    st.markdown("### 🎯 Most Important Entities (by PageRank)")
    pagerank = nx.pagerank(G)
    top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:15]
    
    pr_df = pd.DataFrame(top_pagerank, columns=['Entity', 'Importance Score'])
    pr_df['Importance Score'] = pr_df['Importance Score'].apply(lambda x: f"{x:.6f}")
    st.dataframe(pr_df, use_container_width=True)

# TAB 5: Data Export
with tab5:
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Entity Data")
        csv = nodes_df.to_csv(index=False)
        st.download_button(
            label="Download Entities CSV",
            data=csv,
            file_name="nasa_entities.csv",
            mime="text/csv"
        )
    
    with col2:
        st.markdown("### Relationship Data")
        csv = edges_df.to_csv(index=False)
        st.download_button(
            label="Download Relationships CSV",
            data=csv,
            file_name="nasa_relationships.csv",
            mime="text/csv"
        )
    
    st.markdown("### 🔍 Sample Data Preview")
    st.markdown("**Entities:**")
    st.dataframe(nodes_df.head(10), use_container_width=True)
    
    st.markdown("**Relationships:**")
    st.dataframe(edges_df.head(10), use_container_width=True)

st.markdown("---")
st.markdown("*Built with NetworkX, Pandas, and Streamlit*")
