# Bancada local do PAIEC

Onda 0 da nossa participação no Predictive AI Evaluation Competition (workshop do NeurIPS 2026). Prazo de código e relatório: 30/10/2026 23:59 AoE. Nada aqui foi submetido ao Codabench nem ao OpenReview.

## Como rodar

Ambiente: Windows 11, Python 3.12.10 num venv local (o Python 3.14 do sistema não tem wheel de pandas). Nada foi instalado no Python global.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install numpy pandas pyarrow scikit-learn scipy huggingface_hub
# versões testadas: numpy 2.5.3, pandas 3.0.6, pyarrow 25.0.1, scikit-learn 1.9.1, scipy 1.18.1, huggingface_hub 2.1.1

# baseline oficial (fica fora do git; usamos o commit 82d330d)
git clone https://github.com/aims-foundations/paiec_baseline

# dados: measurement-db é gated. Peça acesso com sua conta HF, crie um token read
# e salve em paiec/.hf_token (uma linha). Embeddings públicos não precisam de token.
.venv/Scripts/python download_data.py          # ~70 MB, sem traces
.venv/Scripts/python describe_data.py          # resumo em results/data_summary.json

# avaliador
.venv/Scripts/python eval_local.py --model models/constant --seed 0
.venv/Scripts/python eval_local.py --model models/global_mean --seed 0
.venv/Scripts/python eval_local.py --model models/official_empirical_mean --seed 0
.venv/Scripts/python eval_local.py --model models/beta_pair --seed 0 --strict   # confere o contrato do teste
```

Sem acesso aos dados, `make_synthetic.py` gera 6 benchmarks falsos (IRT 2PL) com o mesmo layout, e `--data data/synthetic` roda tudo.

Para `run_seeds`: `.venv/Scripts/python eval_local.py --model models/X --seed {seed} --out results/X_{seed}.json`. O JSON tem `brier_alc`, `B0` a `B31`, `labeled_scope`, `per_fold`, `worst_fold`, `brier_alc_bench_macro`, `per_benchmark`, `n_pairs` e `seconds`; a última linha do stdout é o mesmo JSON.

## O avaliador (`eval_local.py`)

Ele segue o protocolo do site e de `paiec_baseline/tools/streaming_ingestion.py`:

- par = (subject_id, benchmark) com pelo menos 80 itens distintos;
- os itens do par se dividem 50/50, ao acaso, em pool de aquisição e pool de avaliação. Só o pool de avaliação é pontuado;
- o pool de aquisição chega item a item. Com `labeling.py`, o avaliador chama `predict` no candidato e depois `acquisition_function(input, prediction, labeled, context)`, que devolve bool. Sem `labeling.py`, vale a política padrão oficial, amostragem seletiva uniforme com P(comprar) = labels_remaining / items_remaining, que sempre fecha 31 rótulos;
- o pool de avaliação é pontuado com 0, 1, 3, 7, 15 e 31 rótulos da mesma trajetória aninhada. Por padrão `labeled` só traz rótulos do par; com `--labeled-scope benchmark` traz também os dos outros pares do benchmark (seção abaixo);
- Brier ALC = 0,1·B0 + 0,2·B1 + 0,2·B3 + 0,2·B7 + 0,2·B15 + 0,1·B31, média simples entre pares. Também gravo a média por benchmark (`brier_alc_bench_macro`);
- a divisão do pool, a ordem do fluxo e a aquisição aleatória dependem só de (seed, par), então todos os modelos veem os mesmos itens na mesma ordem;
- o `benchmark_id` vira um hash anônimo, e todos os campos são strings, com "" para faltante;
- quando um sujeito tem várias tentativas no mesmo item, sorteio uma (com a seed). O teste pontua respostas 0/1 isoladas, não médias.

Divisões (`--split`):

- `lobo` (padrão): deixa um benchmark de fora, treina nos outros 3 elegíveis e pontua os pares do quarto. É a que imita o teste oculto, porque o site diz que "benchmarks are randomly assigned to separate training and test pools before curation". Todo benchmark do teste é novo;
- `pairs`: segura 20% dos pares sujeito-benchmark. Otimista, porque o benchmark já foi visto;
- `subjects`: segura 20% dos `normalized_name`. Serve para medir o quanto o modelo depende de conhecer o sistema.

Contrato do modelo: um diretório com `model.py` (`predict`) e, se quiser, `labeling.py`. Se `model.py` tiver `local_fit(train_obs, subjects, items)`, o avaliador chama essa função uma vez por fold, num módulo novo, só com as linhas de treino do fold. Na submissão real, o mesmo modelo carrega parâmetros pré-treinados de arquivos no zip. Com `predict_batch(inputs, labeled)` o avaliador faz uma chamada por checkpoint.

## Escopo de `labeled` (`--labeled-scope`)

O site oficial diz que "labeled contains [[subject, item], label] entries across sampled subject–benchmark pairs", e que as contagens do `context` valem para o par atual. O BLE oficial filtra `labeled` por sujeito, e o empirical mean filtra por sujeito e benchmark. Ou seja, no teste o modelo recebe rótulos de outros sistemas, o que permite aprender online a dificuldade dos itens do benchmark novo.

- `--labeled-scope pair` (padrão, compatível com os números anteriores): `labeled` só tem os rótulos do par.
- `--labeled-scope benchmark`: os pares de um benchmark (no mesmo fold e seed) andam juntos, checkpoint a checkpoint. No checkpoint k, cada par primeiro completa k rótulos próprios. As chamadas de aquisição desse trecho veem os próprios rótulos e os rótulos dos outros pares no checkpoint anterior. A avaliação em k vê os k rótulos do par e os k primeiros rótulos de cada outro par do benchmark. A trajetória é a mesma aninhada do modo `pair`; a ordem das entradas é embaralhada com uma semente de (seed, par, checkpoint), e o formato continua `[[subject, item], label]`.

O JSON grava `labeled_scope` e `mean_labels_visible_at_31` (no LOBO real, 1.851,6 rótulos visíveis em média no checkpoint 31). Os baselines filtram pelo par e dão o mesmo score nos dois modos, o que serve de conferência.

```bash
.venv/Scripts/python eval_local.py --model models/X --seed 0 --labeled-scope benchmark
```

Tempo, seed 0: com aquisição aleatória, de 26 a 35 s, contra ~22 s no modo `pair`. Com `labeling.py`, 243 s, porque cada candidato chama `predict` e `acquisition_function` com cópias de ~1.850 rótulos, como faz o cliente oficial. Um `predict` que percorre `labeled` inteiro a cada item paga isso em todo checkpoint; indexe por sujeito e benchmark uma vez por chamada, ou memorize por `id` da lista (a lista é a mesma em todas as chamadas de um checkpoint).

Desde esta versão o avaliador faz uma cópia de `labeled` por checkpoint, compartilhada entre os itens, e rejeita o modelo que alterar a lista. O cliente oficial copia a cada chamada; o resultado só muda para modelos que mutam `labeled`, e esses agora falham.

O site também diz que cada avaliação formativa amostra pares de vários benchmarks. Não sei se `labeled` mistura benchmarks diferentes; o modo `benchmark` não mistura.

## Contrato estrito (`--strict`)

O cliente oficial (`paiec_baseline/tools/streaming_ingestion.py`) chama `predict(input, labeled)` item a item, em workers recriados a cada checkpoint. `predict_batch` não existe no teste. `--strict` confere isso numa amostra de pares (`--strict-frac`, padrão 0,05, no mínimo 1 par por fold), em todos os checkpoints desses pares:

1. se houver `predict_batch`, compara com `[predict(x, labeled) for x in inputs]`. Diferença acima de 1e-9 é falha;
2. replay de worker novo: logo depois do `local_fit`, o avaliador guarda um `deepcopy` das variáveis globais do módulo (sem funções, classes e módulos). Em cada checkpoint conferido, restaura essa cópia, apaga as globais criadas depois, refaz as previsões item a item em ordem inversa e compara com o que foi pontuado. Depois devolve o estado vivo. Diferença acima de 1e-9 é falha. Cache que só memoriza (mesma entrada, mesma saída) passa.

O JSON ganha `strict_ok` (true/false; null sem a flag), `strict_pairs_checked` e `strict_failures` (até 20 mensagens). Com falha, o avaliador grava o JSON, imprime `STRICT FAILED ...` no stderr e sai com código 2.

```bash
.venv/Scripts/python eval_local.py --model models/beta_pair --seed 0 --strict
```

Custo medido em `beta_pair`, seed 0, conferindo 13 de 140 pares: no modo `pair`, ~22 s sem a flag e 36 s com ela; no modo `benchmark`, 35 s sem e 63 s com. O extra cresce com o custo de `predict`: cada par conferido roda as previsões mais uma ou duas vezes.

O que ele não pega: estado guardado em outros módulos importados pelo `model.py`, em disco ou em variáveis de ambiente, e modelos que só funcionam depois de `local_fit` (no teste não há `local_fit`; os parâmetros precisam vir de arquivos do zip).

## Teto de pares, folds de estresse e latência

**`--max-pairs-per-bench N`.** Sorteia no máximo N pares por benchmark em cada fold (semente = `--seed`). Cada avaliação formativa do site tem no máximo 1.000 pares sujeito-item, uns 10 pares sujeito-benchmark, então `--max-pairs-per-bench 10` imita esse tamanho. Com `--labeled-scope benchmark`, só os pares sorteados entram em `labeled`: com N = 10 são 310 rótulos visíveis no checkpoint 31, contra 2.542 em multi_swebench sem teto. A opção existia desde o início, mas quebrava no pandas 3 (`groupby.apply` descartava a coluna `bench`). Foi corrigida no commit desta seção.

**`--stress`.** Acrescenta dois folds extras deixados de fora, matharena e mmdocrag, treinando nos 4 benchmarks binários. A binarização é `response >= 0,5` numa tentativa sorteada. Em matharena, 99,9% das tentativas já são 0 ou 1; as notas fracionárias de provas (0,1%) caem no corte. Em mmdocrag, a resposta é uma nota contínua em passos de 0,04 com média 0,67; o corte dá um benchmark de taxa alta (0,78 nos pares avaliados), que é justamente o caso de estresse para priors aprendidos em benchmarks de taxa baixa. Cada linha de mmdocrag vem de uma de 4 `test_condition`; o sorteio da tentativa escolhe uma delas ao acaso. Esses benchmarks usam um gerador de números próprio, então ligar `--stress` não muda os números dos folds normais. O JSON ganha `stress.folds` (resumo por fold), `stress.worst_fold_incl_stress` e `stress.worst_brier_alc_incl_stress`; `brier_alc` e `brier_alc_bench_macro` continuam só com os folds normais.

Seed 0, LOBO, folds de estresse (ALC):

| modelo | matharena (81 pares, taxa 0,64) | mmdocrag (68 pares, taxa 0,78) |
| --- | --- | --- |
| constante 0,5 | 0,250 | 0,250 |
| baseline oficial | 0,256 | 0,183 |
| `beta_pair` | 0,271 | 0,254 |

O `beta_pair` encolhe para a taxa de treino (~0,2) e perde para 0,5 nos dois. É o efeito que o fold de estresse existe para mostrar.

**Por fold e macro.** Todo JSON agora tem `per_fold` (no LOBO, igual a `per_benchmark`), `worst_fold` e `worst_fold_brier_alc`, além de `brier_alc_bench_macro`, que já existia.

**`--latency N`.** Depois da avaliação, o avaliador escolhe o benchmark com mais pares (multi_swebench), roda `local_fit` sem ele e leva todos os pares a 31 rótulos com a política aleatória. Depois mede N chamadas reais de `predict(input, labeled)`, item a item, sem `predict_batch`, com `labeled` no tamanho do modo benchmark (respeita `--max-pairs-per-bench`). O tempo cobre só a chamada, sem as cópias. O JSON ganha `latency` com `p50_ms`, `p95_ms`, `mean_ms`, `first_call_ms`, `labeled_size_mean` e `projected_hours_220k_calls` (média × 220 mil). Com 2.000 chamadas e 2.542 rótulos, o `beta_pair` deu p50 0,33 ms, p95 0,53 ms e projeção de 0,02 h.

```bash
.venv/Scripts/python eval_local.py --model models/X --seed 0 --labeled-scope benchmark \
    --max-pairs-per-bench 10 --stress --latency 2000 --strict
```

Custo: `--stress` soma uns 20 s com os baselines (149 pares a mais); `--latency 2000` leva poucos segundos com modelos rápidos, mais N × o tempo de `predict` com os lentos.

## Os dados

Fonte: `aims-foundations/measurement-db` no Hugging Face (revisão de 18/09/2026, schema v3). São 6 benchmarks: matharena, mmdocrag, multi_swebench, real_webagents, researchcodebench e swe_rebench. O README do dataset fala em 571.921 observações. As tabelas que usamos (benchmarks, subjects, items, response) somam cerca de 70 MB; os traces somam ~1 GB e o `raw/`, 10,4 GB. Nada disso entra no git.

Tabelas: `subjects` (subject_id e os 8 campos do sistema), `items` (item_id, benchmark_id, content, item_features, grading_criterion, verifier...), `response` (subject_id, item_id, trial, test_condition, interactors, response em float). O `interactors` do input vem da linha de resposta.

O repositório público `measurement-db-embed` (text-embedding-3-small, 1536 dims) só cobre 3 benchmarks: real_webagents (233 itens), researchcodebench (212) e swe_rebench (6.271). Os embeddings são de 13/09, antes da migração de 18/09 que mudou os IDs de item: batem 100% pelo sha1 do conteúdo e 0% pelo `item_id`.

O que existe de fato (de `describe_data.py`, revisão `bc8204d`):

| benchmark | elegível | sujeitos | itens | obs. sujeito-item | taxa de acerto | pares ≥80 itens | texto mediano (chars) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| matharena | não (response_type mixed, até 200 tentativas) | 104 | 1.755 | 20.354 | 0,70* | 81 | 521 |
| mmdocrag | não (fração 0..1 em passos de 0,04) | 68 | 2.000 | 134.498 | 0,78* | 68 | 112 |
| multi_swebench | sim | 82 | 2.126 | 57.808 | 0,148 | 82 | 1.147 |
| real_webagents | sim | 33 | 233 | 3.759 | 0,373 | 26 | 92 |
| researchcodebench | sim | 31 | 212 | 6.572 | 0,353 | 31 | 96.288 |
| swe_rebench | sim | 1 | 6.306 | 6.306 | 0,471 | 1 | 857 |

\* binarizado em ≥ 0,5, só para descrição.

- 571.921 linhas de resposta e 229.297 observações sujeito-item depois de sortear uma tentativa. 287 subject_ids, 133 `normalized_name` distintos, 34 deles em 2 ou mais benchmarks. 12.632 itens.
- Só 4 benchmarks são binários com granularidade de item, o critério do `prepare_data.py` oficial. Eles dão **140 pares** avaliáveis, e 82 deles são de multi_swebench. swe_rebench tem um único sujeito (Qwen3-Coder 480B no OpenHands, 34 tentativas por item). matharena e mmdocrag não entram no avaliador, mas servem de treino auxiliar para quem carregar com `bench_data.load(..., eligible_only=False)`, já que nunca ficam de fora no LOBO.
- `item_features` é texto `chave=valor;...`, não JSON: `lang=c++` (multi_swebench), `website=staynb` (real_webagents), `paper=advantage-alignment` (researchcodebench), `competition=aime_2025;problem_idx=2` (matharena). swe_rebench e mmdocrag não têm features.
- `interactors` está vazio em todas as linhas. `harness_version`, `access_date` e `reasoning_effort` quase sempre também. Os campos úteis do sistema são `normalized_name`, `provider`, `release_date` e, às vezes, `harness`.
- O texto varia muito: researchcodebench tem mediana de 96 mil caracteres por item (21 MB no total), real_webagents de 92.
- Os embeddings públicos batem 100% pelo sha1 do conteúdo e 0% pelo `item_id`. Junte pelo hash.
- Teste oculto: o site diz que os benchmarks foram divididos ao acaso entre treino e teste antes da curadoria. Todo benchmark do teste é novo. Sistemas devem se repetir em parte (são os mesmos fornecedores e modelos de 2024-2025), mas isso não está escrito em lugar nenhum.

## Números dos baselines (LOBO, seeds 0, 1 e 2)

| modelo | Brier ALC (média de pares) | macro por benchmark | B0 | B1 | B3 | B7 | B15 | B31 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) constante 0,5 | 0,2500 | 0,2500 | 0,250 | 0,250 | 0,250 | 0,250 | 0,250 | 0,250 |
| (b) taxa média global do treino | 0,2248 ± 0,0001 | 0,2602 | 0,225 | 0,225 | 0,225 | 0,225 | 0,225 | 0,225 |
| (c) baseline oficial (empirical mean) | 0,2213 ± 0,0016 | 0,2608 | 0,250 | 0,317 | 0,215 | 0,188 | 0,176 | 0,171 |
| sonda: Beta-Binomial (`beta_pair`) | 0,1923 ± 0,0011 | 0,2251 | 0,225 | 0,211 | 0,195 | 0,183 | 0,175 | 0,170 |
| sonda: Beta-Binomial + aquisição por incerteza do BLE | 0,1936 ± 0,0018 | 0,2240 | 0,225 | 0,214 | 0,196 | 0,185 | 0,176 | 0,170 |

± é o desvio entre seeds. O repositório oficial não publica números de referência, e o BLE não rodou porque exige chave de API. Cada execução leva 35 a 50 s, e uns 30 s disso são a carga dos dados.

O que eu tiro daqui: o baseline oficial perde para 0,5 em B1 (0,317), porque um único rótulo vira previsão 0 ou 1. A taxa global ganha na média de pares só porque multi_swebench (taxa 0,15 a 0,22) domina os pares; na média por benchmark ela perde para 0,5. A sonda Beta-Binomial, que só suaviza a média do par em direção à taxa global, já tira 0,03 do ALC. A aquisição por incerteza não ajudou.

## Cinco ideias de modelo, em ordem

1. **Atualização bayesiana logística por par.** O peso da métrica está 80% em B1..B31, e o baseline oficial perde para a constante 0,5 com 1 rótulo porque não suaviza. Modelo: logit p = θ_par + d_item, com prior em θ_par vindo do modelo de população e posterior por grade ou Laplace a cada rótulo revelado. É barato, roda em milissegundos e é o que mais move a métrica. O Beta-Binomial em `models/beta_pair` já é a versão sem itens dessa ideia.
2. **Prior do sistema a partir de treino e metadados.** Os mesmos `normalized_name` aparecem em vários benchmarks. Uma IRT multibenchmark no treino dá uma habilidade latente por sistema; para sistemas novos, uma regressão em provider, release_date, reasoning_effort e harness. Isso define o θ do par em B0 e o quanto confiar nele. Uma chamada a LLM aprovado (perguntando "quão forte é este sistema?") pode entrar como feature extra.
3. **Dificuldade do item pelo conteúdo.** O benchmark do teste é novo, então não há parâmetros de item. Um regressor de embedding de `item_content` e de `item_features` para a dificuldade relativa dentro do benchmark (b da IRT, centrado) dá o d_item. O modelo de embedding precisa ir no zip ou num dos 5 repositórios; a API aprovada é alternativa. Aqui entra **importance weighting**: os 5 benchmarks de treino não têm a mesma distribuição de itens do benchmark de teste. Pesos por razão de densidade (classificador treino-vs-teste nos embeddings dos itens do pool de avaliação, que o modelo vê no `predict`) corrigem o regressor para a distribuição do benchmark novo.
4. **Aquisição que sabe que enviesa a amostra.** Com `labeling.py` escolhendo itens por incerteza, os rótulos revelados deixam de ser amostra aleatória do par, e a média deles fica enviesada. Ou o modelo condiciona na dificuldade do item (ideia 1+3, e aí o viés some porque a seleção depende só de x), ou se corrige a média com pesos 1/q(x) da política. Vale medir se a aquisição ativa ganha da aleatória: nos dados reais, a política de incerteza do BLE não melhorou o Beta-Binomial (0,1936 contra 0,1923, dentro do ruído entre seeds).
5. **Calibração e adaptação a label shift entre benchmarks.** A taxa de acerto varia muito de benchmark para benchmark, de 0,15 em multi_swebench a 0,47 em swe_rebench. Na média por benchmark, prever a taxa global dá 0,260, pior que 0,5. B0 é previsão sob mudança de prior: dá para estimar a taxa base do benchmark novo a partir da distribuição das previsões nos itens de avaliação (EM de Saerens-Latinne-Decaestecker) e recalibrar, e depois atualizar com os rótulos revelados. A calibração final (temperatura por orçamento) se ajusta por LOBO. É onde o trabalho de data drift do dono do projeto entra de forma mais direta.

## Riscos e pontos em aberto

- `labeled` no teste traz rótulos de outros pares (site oficial; veja "Escopo de `labeled`"). Ainda não sabemos quantos pares por benchmark entram numa avaliação nem se benchmarks se misturam. Confira no starting kit do Codabench, que exige login.
- Cada avaliação formativa usa no máximo 1.000 pares sujeito-item (aquisição + avaliação), ou seja, uns 10 pares sujeito-benchmark. O placar formativo vai ser ruidoso; confie no LOBO local com várias seeds.
- Só 4 benchmarks elegíveis e 140 pares, 82 deles de multi_swebench. O LOBO tem 4 folds, e o de swe_rebench é um par só. Olhe a média de pares e a média por benchmark juntas.
- `local_fit` recebe só os 4 benchmarks binários. matharena e mmdocrag (respostas fracionárias e muitas tentativas) podem dar mais sinal de habilidade dos sistemas, mas precisam de outra regra de binarização.
- O teste roda com workers recriados a cada checkpoint e 8 h no total, inclusive setup. Modelos com chamada de LLM por item precisam de orçamento de tempo.
