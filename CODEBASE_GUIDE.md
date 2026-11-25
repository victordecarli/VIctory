# Guia do código do VIctory

Este guia descreve a implementação presente neste checkout. “Verificado” indica que a afirmação foi confirmada no código-fonte ou na documentação do projeto; “Limite” identifica comportamentos incompletos ou dependentes da plataforma. O objetivo é orientar a navegação e as mudanças no repositório, não afirmar que todas as APIs do console ou todos os shaders são compatíveis.

## Arquitetura e limites entre crates

O executável do jogo roda como código x86-64 nativo dentro do processo do VIctory. O loader mapeia a imagem, o HLE resolve as APIs importadas do console e a camada de GPU implementa a interface D3D12.X sobre o D3D12 do host. O jogo lê seus arquivos em tempo de execução por meio dessas APIs; somente a tradução de shaders é armazenada em cache como dado convertido. Consulte `docs/architecture.md`.

| Crate | Responsabilidade e pontos de entrada úteis |
|---|---|
| `victory-core` | `Config`, logging e `ensure_outside_repo`; configuração compartilhada em tempo de execução e aplicação da política de assets. |
| `victory-pe` | Leitura e análise pura de PE32+ e bibliotecas de importação (`image`, `layout`, `implib`); não mapeia imagens nem faz I/O de arquivos. |
| `victory-formats` | Layout e destiling de texturas, decodificação de BCPack e XMA2. HLE/GPU usam este crate em vez de duplicar código de formatos. |
| `victory-game` | Identificação de builds do jogo e leitura de SGD5 (`identify`, `sgd5`). Conhecimento específico do título deve ficar aqui. |
| `victory-shader` | Decodificação GFX10/SP3, IR, estruturação de fluxo de controle, rastreamento de recursos, emissão de DXIL e validação. |
| `victory-pipeline` | Abstração `Converter` e cache endereçado por conteúdo. Não conhece formatos do jogo. |
| `victory-loader` | Preparação e mapeamento de imagens, vinculação de imports, carregamento de DLLs do título, TLS, threads e relatórios de falha. A maior parte da execução é exclusiva do Windows. |
| `victory-gpu` | Objetos D3D12.X, descritores, listas de comandos, rastreamento de memória/recursos, criação de pipelines e execução no D3D12 do host. Chama a tradução de shaders, mas não implementa sua semântica. |
| `victory-hle` | Exports das APIs do console, agrupados em `modules`; objetos de runtime e política de passagem para APIs do host. Encaminha GPU para `victory-gpu` e decodificação para `victory-formats`. |
| `apps/victory` | Apenas a interface de linha de comando: inspect, census, shader-info, translate, run e config. |
| `xtask` | Verificações da política de assets e de citações, além da configuração de hooks. |

A direção das dependências deve seguir dos componentes de nível mais alto para os mais básicos: app → HLE; HLE → GPU/loader/PE/formats; GPU → shader/pipeline/formats; loader → core/PE; pipeline → core. Crates puros devem continuar portáveis e sem dependências de I/O, processos ou APIs do host. Para compartilhar uma função, não crie uma dependência no sentido inverso: coloque a lógica no crate inferior responsável por ela.

## Entradas e fluxos de execução

`apps/victory/src/main.rs` define os comandos Clap em `Command` e os encaminha a partir de `main()`. `inspect`, `census`, `shader` e `translate` são comandos de análise. `translate` percorre arquivos para diagnóstico; não preenche o cache de runtime. `run`, em `apps/victory/src/run.rs`, é condicionado à plataforma Windows.

Fluxo de `run` verificado: lê os bytes do executável → `victory_loader::prepare` → identifica o hash via `victory_game::identify` → configura os caches de imagem e shader → `victory_loader::load` → configura a identidade do título no HLE, o armazenamento, as unidades virtuais G:/T: e o modelo do console → carrega DLLs do título que ficam ao lado do executável → vincula imports da EXE e das DLLs usando `HleResolver` envolvido por `WithTitleDlls` → pode parar após a vinculação → instala o relatório de falhas e o callback para DLLs carregadas tardiamente → chama `spawn_main` de forma `unsafe`, depois que os imports foram vinculados → aguarda a thread do título. A CLI e o README mostram o comando `victory run [exe] --execute`; sem caminho, `run` usa o único `.exe` na raiz do `game_dir` configurado.

A ordem de resolução de imports, implementada pelo resolver e pela vinculação do HLE, é: override do VIctory → export do host permitido → DLL local do título → stub gerado para import não implementado. `--strict` desativa a passagem para APIs do host; `--unimplemented trap|zero` define o comportamento dos imports ausentes. `victory census` classifica imports sem mapear DLLs do título; `run` sem `--execute` mapeia e vincula, mas não inicia o código do jogo.

## Pipelines críticos

### Carregamento do executável e memória

`victory-pe` analisa os metadados; `victory-loader::prepare` prepara os dados da imagem; `load` a mapeia e produz um `LoadedImage`; `bind_imports` ajusta os destinos das chamadas de imports conforme o resolver. Os módulos do loader `resolve.rs`, `tls.rs`, `threads.rs`, `thunks.rs` e `crash.rs` cuidam, respectivamente, da resolução, do TLS, do suporte a threads, dos thunks gerados e dos relatórios de falhas. O TLS da imagem principal é tratado de forma diferente do TLS de DLLs (slot 0 para a imagem principal e outro slot para DLLs); preserve essa distinção ao alterar o mapeamento. Os relatórios de falhas identificam `module+rva`, conforme a documentação de arquitetura.

A memória do jogo é memória do processo; não é uma cópia isolada em um sandbox de emulação. O rastreamento de memória e recursos da GPU conecta os endereços visíveis ao console aos recursos D3D12 do host. Por isso, validade e duração de ponteiros, além da segurança de FFI, são pontos centrais: caminhos alcançáveis pelo jogo devem retornar um código de falha da API ou descartar o trabalho, em vez de entrar em panic; dereferências de ponteiros crus precisam ficar em funções corretamente marcadas como `unsafe` e ter invariantes de segurança explícitos.

### HLE e APIs do host

`crates/victory-hle/src/modules/mod.rs` registra os módulos; arquivos como `files.rs`, `dstorage.rs`, `audio.rs`, `d3d12x.rs`, `gameinput.rs` e `xgameruntime.rs` implementam famílias de APIs. `runtime.rs` mantém o estado do título/runtime e `host.rs` define o comportamento de passagem para APIs do host. As APIs de arquivos convertem as unidades virtuais do console em diretórios configurados no host; saves e armazenamento persistente ficam no diretório de dados. A rede é desativada por projeto. Ao adicionar um export, confira tanto a vinculação de imports quanto os caminhos `LoadLibrary`/`GetProcAddress` e mantenha a política consistente.

### Tradução de shaders e cache

Fluxo de tradução verificado: container do console → `victory_shader::ir::build` (decodificação nativa para SSA) → `structure` (conversão do fluxo de controle EXEC/lanes) → `resource` (proveniência de descritores e endereços) → `emit::translate` (DXIL e metadados do host). Os módulos centrais são `gfx10`, `native`, `ir`, `structure`, `resource` e `emit`.

Em runtime, `victory-gpu/src/shader_cache.rs` chama `emit::translate` durante a criação do pipeline. A chave de entrada do cache inclui uma impressão digital do executável/configurações do VIctory e os bytes do container; `HostShader` é serializado no formato `VSH1`. Ao ampliar `HostShader`, atualize tanto `encode` quanto `decode`. `victory-pipeline::Converter` exige `id` estável, incremento de `version` quando a semântica do resultado mudar e conversão determinística. Falhas de tradução podem impedir a criação do pipeline ou descartar draws; a cobertura incompleta de shaders é uma limitação conhecida da renderização.

### Integração D3D12

Os exports D3D12.X do HLE chegam a `victory-gpu`. Os módulos `device`, `list`, `queue`, `resource`, `descriptor`, `pipeline`, `draw`, `indirect`, `frame` e `host` separam objetos/comandos do console da execução no host. A criação de pipelines combina metadados dos shaders com a configuração de root signature e pipeline no host. Duração de recursos/descritores e validação de argumentos indirect são pontos de risco elevado. Travamento da GPU ou remoção do dispositivo são falhas conhecidas; opções de diagnóstico (`VICTORY_D3D12_DEBUG`, `GBV`, `DRED`, `WARP`, captura e timing) estão documentadas no README.

## Desenvolvimento e depuração seguros

* Antes de mudar limites entre componentes ou fixtures, leia `docs/architecture.md` e `docs/asset-policy.md`. Nunca adicione ao repositório bytes derivados do jogo, dumps, screenshots, strings ou outros dados proprietários. Testes devem gerar dados sintéticos ou usar caminhos fornecidos por variáveis de ambiente, ignorando-os quando esses caminhos não estiverem definidos.
* Mantenha I/O, logging e APIs do host fora dos crates puros. Condicione o código Windows e suas dependências Windows juntos. A CI Linux exclui HLE e app enquanto as lacunas de portabilidade não forem resolvidas.
* Considere métodos de FFI/tabelas e tudo que eles chamam como alcançável pelo jogo: evite caminhos que possam causar panic, erros descartados, `unsafe` sem justificativa e funções seguras que dereferenciam ponteiros crus. Matches de enums do projeto devem listar as variantes; código que soma ou combina agregados deve destruturar todos os campos.
* Use `VICTORY_LOG=victory_loader=trace,info`; compile os diagnósticos com `cargo build --profile diag -p victory --features diagnostics`. As opções `VICTORY_WATCH`, `VICTORY_STACKS`, `VICTORY_CRASH_SAVE`, `VICTORY_CAPTURE`, validação D3D12, WARP e controles de skip/timing de shaders estão detalhadas no README. Artefatos de runtime devem ir para os diretórios configurados de cache/dados, fora do repositório.
* Adicione testes sintéticos e focados no crate responsável. Probes de integração do loader ficam em `crates/victory-loader/tests`; a proteção contra plantio de DLLs do HLE fica em `crates/victory-hle/tests`. Não crie testes que presumam que o executável de teste é um PE em plataformas que não sejam Windows.
* No Windows, verifique com os comandos de clippy/test/policy do `AGENTS.md`. As verificações portáveis excluem HLE/app, e o clippy exclui GPU por causa de código atualmente sem uso em plataformas específicas. Informe claramente quando não foi possível verificar no Windows. Este guia não altera a implementação.

## Remotes do fork e segurança de branches

A convenção desejada é manter a `main` local acompanhando `origin/main` no fork pessoal; o repositório original deve ser configurado como `upstream` somente para fetch. Envie branches de trabalho ao fork, nunca ao upstream. Estado Git verificado neste checkout: a branch atual é `main` e acompanha `origin/main`; `.git/config` contém apenas `origin`, configurado como `https://github.com/victordecarli/VIctory.git` tanto para fetch quanto para push. Não foi possível confirmar a URL do repositório original a partir do código do projeto; portanto, não a adivinhe nem configure um remote silenciosamente. Quando a URL canônica for conhecida, configure `upstream` para ela e remova sua URL de push (ou configure uma URL que impeça push); mantenha `origin` apontando para o fork do usuário e `main` acompanhando `origin/main`. Antes de qualquer push, confirme que o destino é somente `origin`.
