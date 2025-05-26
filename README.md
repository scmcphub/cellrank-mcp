# cellrank-MCP

Natural language interface for scRNA-Seq analysis with cellrank through MCP.

## 🪩 What can it do?

- IO module like read and write scRNA-Seq data
- Preprocessing module,like filtering, quality control, normalization, scaling, highly-variable genes, PCA, Neighbors,...
- Tool module, like clustering, differential expression etc.
- Plotting module, like violin, heatmap, dotplot

## ❓ Who is this for?

- Anyone who wants to do scRNA-Seq analysis natural language!
- Agent developers who want to call cellrank's functions for their applications

## 🌐 Where to use it?

You can use cellrank-mcp in most AI clients, plugins, or agent frameworks that support the MCP:

- AI clients, like Cherry Studio
- Plugins, like Cline
- Agent frameworks, like Agno 


## 📚 Documentation

scmcphub's complete documentation is available at https://docs.scmcphub.org


## 🎬 Demo

A demo showing scRNA-Seq cell cluster analysis in a AI client Cherry Studio using natural language based on cellrank-mcp



## 🏎️ Quickstart

### Install

Install from PyPI
```
pip install cellrank-mcp
```
you can test it by running
```
cellrank-mcp run
```



#### run cellrank-mcp locally
Refer to the following configuration in your MCP client:

check path
```
$ which cellrank 
/home/test/bin/cellrank-mcp
```

```
"mcpServers": {
  "cellrank-mcp": {
    "command": "/home/test/bin/cellrank-mcp",
    "args": [
      "run"
    ]
  }
}
```

#### run cellrank-server remotely
Refer to the following configuration in your MCP client:

run it in your server
```
cellrank-mcp run --transport shttp --port 8000
```

Then configure your MCP client in local AI client, like this:
```

"mcpServers": {
  "cellrank-mcp": {
    "url": "http://localhost:8000/mcp"
  }
}
```
## 🤝 Contributing

If you have any questions, welcome to submit an issue, or contact me(hsh-me@outlook.com). Contributions to the code are also welcome!

## Citing

If you use cellRank-mcp in for your research, please consider citing  following work: 
> Weiler, P., Lange, M., Klein, M. et al. CellRank 2: unified fate mapping in multiview single-cell data. Nat Methods 21, 1196–1205 (2024). https://doi.org/10.1038/s41592-024-02303-9
