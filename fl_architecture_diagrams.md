# FL Simulation — Architecture Diagrams (Mermaid)

---

## 1. Complete FL System Architecture Overview

```mermaid
flowchart TD
    subgraph DATASET["Dataset Layer"]
        D1[(CIC-DDoS2019\n426,076 samples · 11 attack types\n77 raw network flow features)]
        D2[MinMaxScaler Normalization]
        D3["5-Method Hybrid Feature Selection\nL1-SVC · RF Importance · ANOVA · Chi² · Mutual Info"]
        D4[65 Features Selected\nUnion of all methods]
        D1 --> D2 --> D3 --> D4
    end

    subgraph SPLIT["Non-IID Data Partitioning + SMOTE"]
        S1["Client 1 Partition\n80% Normal / 20% DDoS\nSMOTE → Balanced"]
        S2["Client 2 Partition\n20% Normal / 80% DDoS\nSMOTE → Balanced"]
        D4 --> S1 & S2
    end

    subgraph SERVER["FL Server · Flower Framework · gRPC · Port varies per model"]
        SV1([Initialize\nGlobal Model θ₀])
        SV2[Aggregate Parameters\nStrategy-specific per model]
        SV3["Evaluate θ_r on Test Set\n63,912 samples · 14,206 Normal · 49,706 DDoS"]
        SV4{F1 Macro\n> best so far?}
        SV5[(Best Round\nModel Checkpoint)]
        SV6[Save Results\nHistory CSV · Round Plot · Confusion Matrix]
    end

    subgraph C1["Client 1 · Docker Container"]
        C1R[Receive Global Params θ_r]
        C1T[Local Model Training\nclass_weight per local distribution]
        C1S[Send θ_c1 · num_samples · loss]
        C1R --> C1T --> C1S
    end

    subgraph C2["Client 2 · Docker Container"]
        C2R[Receive Global Params θ_r]
        C2T[Local Model Training\nclass_weight per local distribution]
        C2S[Send θ_c2 · num_samples · loss]
        C2R --> C2T --> C2S
    end

    S1 --> C1T
    S2 --> C2T
    D4 -->|"held-out test set"| SV3

    SV1 -->|"Broadcast θ_r"| C1R & C2R
    C1S & C2S -->|"Local params + metrics"| SV2
    SV2 --> SV3
    SV3 --> SV4
    SV4 -->|Yes| SV5 --> SV1
    SV4 -->|No| SV1
    SV5 --> SV6
```

---

## 2. Data Preprocessing & Feature Selection Pipeline

```mermaid
flowchart LR
    A[(CIC-DDoS2019\nunified_train.csv\nunified_val.csv\nunified_test.csv)]

    subgraph PRE["Preprocessing"]
        direction TB
        P1[Merge train + val\nignore_index=True]
        P2[Select numeric columns only]
        P3[Remove duplicate rows]
        P4[Remove zero-variance features\nvar < 1e-6]
        P5[MinMaxScaler\nfit on train · transform all]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph FS["5-Method Hybrid Feature Selection"]
        direction TB
        F1["L1-LinearSVC\nC=0.1 · penalty=l1\n→ non-zero coef features"]
        F2["Random Forest\n100 trees · max_depth=10\n→ cumulative 95% importance"]
        F3["ANOVA F-test\np < 0.01 / n_features\nBonferroni corrected"]
        F4["Chi-square Test\nKBinsDiscretizer 20 bins\np < 0.01"]
        F5["Mutual Information\ntop 75th percentile\n10% subsample for speed"]
        F6{{"UNION\nof all selected"}}
        F1 & F2 & F3 & F4 & F5 --> F6
    end

    subgraph SPLIT["Non-IID Split + SMOTE"]
        direction TB
        SP{"Split by class ratio"}
        SP --> C1P["Client 1\n80% Benign · 20% DDoS\n~119,577 samples"]
        SP --> C2P["Client 2\n20% Benign · 80% DDoS\n~237,094 samples"]
        C1P --> SM1["SMOTE k=5\nrandom_state=42\nClient 1 Balanced"]
        C2P --> SM2["SMOTE k=5\nrandom_state=42\nClient 2 Balanced"]
    end

    A --> PRE --> FS --> SPLIT
```

---

## 3. Generic FL Training Loop (Sequence Diagram)

```mermaid
sequenceDiagram
    participant S  as FL Server
    participant C1 as Client 1
    participant C2 as Client 2

    Note over S: Initialize global model θ₀
    Note over S: Request initial params from random client

    loop Round r = 1 to R
        S  ->>  C1: Broadcast global params θ_{r-1}
        S  ->>  C2: Broadcast global params θ_{r-1}

        activate C1
        Note over C1: Set weights to θ_{r-1}
        Note over C1: Local training on X_c1, y_c1
        C1 -->> S : Return θ_c1, n_c1, loss_c1
        deactivate C1

        activate C2
        Note over C2: Set weights to θ_{r-1}
        Note over C2: Local training on X_c2, y_c2
        C2 -->> S : Return θ_c2, n_c2, loss_c2
        deactivate C2

        Note over S: Aggregate: θ_r = Strategy(θ_c1, θ_c2)
        Note over S: Evaluate θ_r on 63,912 test samples
        Note over S: Track best round by F1 Macro
    end

    Note over S: Restore best-round model
    Note over S: Save CSV · Round Plot · Confusion Matrix
```

---

## 4. CNN 1D — FedAvg with EarlyStopping + Class Weighting

```mermaid
flowchart TD
    subgraph MODEL["CNN 1D Architecture (per client)"]
        direction LR
        M1["Input\n65 × 1"] --> M2["Conv1D-64\nReLU · same padding"] --> M3[BatchNorm]
        M3 --> M4["Conv1D-64\nReLU · same padding"] --> M5[BatchNorm]
        M5 --> M6[MaxPooling1D] --> M7["Conv1D-128\nReLU · same padding"] --> M8[BatchNorm]
        M8 --> M9[GlobalAvgPool] --> M10["Dense-128\nReLU"] --> M11["Dropout 0.4"]
        M11 --> M12["Dense-1\nSigmoid"]
    end

    subgraph CLOOP["FL Round r (× 10 rounds)"]
        direction TB

        SRV["FL Server\nFedAvg: θ_r = Σ n_i/N · θ_i"]

        subgraph TR1["Client 1 · 80% Normal / 20% DDoS"]
            T1A[Set weights = θ_{r-1}]
            T1B["Compute class_weight\n{0: n/2·n_neg, 1: n/2·n_pos}"]
            T1C["model.fit\nEpochs=5 · batch=128\nEarlyStopping patience=3"]
            T1D[Return θ_c1 · n_c1 · loss]
            T1A --> T1B --> T1C --> T1D
        end

        subgraph TR2["Client 2 · 20% Normal / 80% DDoS"]
            T2A[Set weights = θ_{r-1}]
            T2B["Compute class_weight\n{0: n/2·n_neg, 1: n/2·n_pos}"]
            T2C["model.fit\nEpochs=5 · batch=128\nEarlyStopping patience=3"]
            T2D[Return θ_c2 · n_c2 · loss]
            T2A --> T2B --> T2C --> T2D
        end

        EVAL["Server Evaluate\npredict on 63,912 test samples\nthreshold = 0.5\nrecord acc · prec · rec · f1_macro"]
        BEST{F1 Macro\n> best?}
        CKPT[(Save best\nweights)]

        SRV -->|"broadcast θ_{r-1}"| T1A & T2A
        T1D & T2D -->|"weighted average"| SRV
        SRV --> EVAL --> BEST
        BEST -->|Yes| CKPT
    end

    subgraph OUT["After Round 10"]
        O1[Restore best-round weights]
        O2[Confusion Matrix PNG]
        O3[cnn_server_history.csv]
        O4[Round metrics plot PNG]
    end

    MODEL -.->|"architecture used by both clients"| CLOOP
    BEST -->|No / next round| SRV
    CKPT --> O1 --> O2 & O3 & O4
```

---

## 5. Decision Tree — FedBest Strategy

```mermaid
flowchart TD
    subgraph MODEL["Decision Tree Config"]
        direction LR
        DT["DecisionTreeClassifier\ncriterion=gini · max_depth=12\nmin_samples_leaf=5 · random_state=42"]
    end

    subgraph ROUND["FL Round r (× 5 rounds)"]
        direction TB

        subgraph C1["Client 1 · 80% Normal"]
            C1A[Retrain DT from scratch\non X_c1, y_c1]
            C1B[Compute train accuracy]
            C1C["Serialize via pickle\nReturn bytes · n_c1 · loss_c1"]
            C1A --> C1B --> C1C
        end

        subgraph C2["Client 2 · 80% DDoS"]
            C2A[Retrain DT from scratch\non X_c2, y_c2]
            C2B[Compute train accuracy]
            C2C["Serialize via pickle\nReturn bytes · n_c2 · loss_c2"]
            C2A --> C2B --> C2C
        end

        AGG["FedBest Aggregation\nSelect client with min loss\nbest_model = argmin_i loss_i"]
        EVAL["Server Evaluate\npredict on test set\nrecord metrics"]

        C1C & C2C --> AGG --> EVAL
    end

    subgraph NOTE["Convergence Note"]
        N1["Both clients retrain from scratch\nwith fixed random_state=42\non the same local data each round\n→ Identical model every round\n→ Same client wins every round\n→ Identical metrics rounds 1–5"]
    end

    subgraph OUT["Results"]
        O1[Confusion Matrix PNG]
        O2[dt_server_history.csv]
    end

    MODEL -.->|"used by both clients"| ROUND
    EVAL --> NOTE
    EVAL --> OUT
```

---

## 6. Random Forest — FederatedForest Strategy

```mermaid
flowchart TD
    subgraph MODEL["Random Forest Config (per client)"]
        direction LR
        RF["RandomForestClassifier\nn_estimators=100 · max_features=sqrt\nmax_depth=None · random_state=42"]
    end

    subgraph ROUND["FL Round r (× 5 rounds)"]
        direction TB

        subgraph C1["Client 1 · 80% Normal"]
            C1A[Retrain RF from scratch\non X_c1, y_c1]
            C1B["Produce 100 decision trees\nestimators_ list"]
            C1C["Serialize via pickle\nReturn bytes · n_c1 · loss"]
            C1A --> C1B --> C1C
        end

        subgraph C2["Client 2 · 80% DDoS"]
            C2A[Retrain RF from scratch\non X_c2, y_c2]
            C2B["Produce 100 decision trees\nestimators_ list"]
            C2C["Serialize via pickle\nReturn bytes · n_c2 · loss"]
            C2A --> C2B --> C2C
        end

        subgraph AGG["FederatedForest Aggregation"]
            AGG1[Deserialize both client forests]
            AGG2["Collect all trees\nall_trees = c1.estimators_ + c2.estimators_"]
            AGG3["Build merged RF\nn_estimators = 200 trees\nDiversity from Non-IID partitions"]
            AGG1 --> AGG2 --> AGG3
        end

        EVAL["Server Evaluate Merged Forest\n200-tree ensemble prediction on test set"]

        C1C & C2C --> AGG --> EVAL
    end

    subgraph OUT["Results"]
        O1[Confusion Matrix PNG]
        O2[rf_server_history.csv]
        O3["Best model = highest F1 Macro round\nsaved via pickle deep copy"]
    end

    MODEL -.->|"100 trees per client"| ROUND
    EVAL --> OUT
```

---

## 7. Logistic Regression — FedAvg on Linear Parameters

```mermaid
flowchart TD
    subgraph MODEL["Logistic Regression Config"]
        direction LR
        LR["LogisticRegression\npenalty=l2 · solver=liblinear\nC=1.0 · max_iter=1000 · random_state=42"]
        INIT["init_model: dummy fit on 2 samples\nto create coef_ and intercept_ arrays"]
    end

    subgraph ROUND["FL Round r (× 10 rounds)"]
        direction TB

        subgraph C1["Client 1 · 80% Normal"]
            C1A["Set model params\ncoef_ = θ_r[0].reshape(1,-1)\nintercept_ = θ_r[1]"]
            C1B["model.fit X_c1, y_c1\nwarm-start from global params"]
            C1C["Return coef_.flatten() · intercept_\nn_c1 · loss"]
            C1A --> C1B --> C1C
        end

        subgraph C2["Client 2 · 80% DDoS"]
            C2A["Set model params\ncoef_ = θ_r[0].reshape(1,-1)\nintercept_ = θ_r[1]"]
            C2B["model.fit X_c2, y_c2\nwarm-start from global params"]
            C2C["Return coef_.flatten() · intercept_\nn_c2 · loss"]
            C2A --> C2B --> C2C
        end

        AGG["FedAvg Aggregation\nθ_r = n_c1/(n_c1+n_c2) · θ_c1\n     + n_c2/(n_c1+n_c2) · θ_c2"]
        EVAL["Server Evaluate\npredict on test set\nrecord metrics"]
        BEST{F1 Macro\n> best?}
        CKPT["Save best params\n[coef_.copy(), intercept_.copy()]"]

        C1C & C2C --> AGG --> EVAL --> BEST
        BEST -->|Yes| CKPT
        BEST -->|No| AGG
    end

    subgraph NOTE["Convergence Note"]
        N1["LR has convex loss surface\nGlobal optimum reached in Round 1\nZero gradient in rounds 2–10\nIdentical metrics → true convergence"]
    end

    subgraph OUT["Results"]
        O1[Confusion Matrix PNG]
        O2[lr_server_history.csv]
    end

    MODEL --> ROUND
    EVAL --> NOTE
    CKPT --> OUT
```

---

## 8. Naive Bayes — FedAvg on Gaussian Statistics

```mermaid
flowchart TD
    subgraph MODEL["Gaussian Naive Bayes Config"]
        direction LR
        NB["GaussianNB\nno hyperparameters\nfit on dummy 2 samples to init structure"]
        PARAMS["Parameters: theta_ class means\n         var_   class variances\n         class_prior_"]
    end

    subgraph ROUND["FL Round r (× 5 rounds)"]
        direction TB

        subgraph C1["Client 1 · 80% Normal"]
            C1A["Set model:\ntheta_ · var_ · class_prior_\nfrom global params"]
            C1B["model.fit X_c1, y_c1\nRecompute Gaussian statistics\nfrom local data"]
            C1C["Return theta_.flatten()\nvar_.flatten() · class_prior_\nn_c1 · loss"]
            C1A --> C1B --> C1C
        end

        subgraph C2["Client 2 · 80% DDoS"]
            C2A["Set model:\ntheta_ · var_ · class_prior_\nfrom global params"]
            C2B["model.fit X_c2, y_c2\nRecompute Gaussian statistics\nfrom local data"]
            C2C["Return theta_.flatten()\nvar_.flatten() · class_prior_\nn_c2 · loss"]
            C2A --> C2B --> C2C
        end

        AGG["FedAvg Aggregation\nθ_r = weighted average of\nclass means · variances · priors"]
        EVAL["Server Evaluate\npredict on test set\nrecord metrics"]
        BEST{F1 Macro\n> best?}
        CKPT["Save best params\ntheta · var · prior copies"]

        C1C & C2C --> AGG --> EVAL --> BEST
        BEST -->|Yes| CKPT
        BEST -->|No| AGG
    end

    subgraph OUT["Results"]
        O1[Confusion Matrix PNG]
        O2[nb_server_history.csv]
    end

    MODEL --> ROUND
    CKPT --> OUT
```

---

## 9. SVM — FedEnsemble (Weighted Soft-Voting)

```mermaid
flowchart TD
    subgraph MODEL["SVM Config (per client)"]
        direction LR
        SVM["SVC\nkernel=rbf · C=10 · gamma=scale\nrandom_state=42 · probability=True"]
        NOTE["probability=True enables predict_proba\nvia Platt scaling — very slow on large data\nO(n²) training complexity"]
    end

    subgraph ROUND["FL Round r (× 5 rounds)"]
        direction TB

        subgraph C1["Client 1 · 80% Normal"]
            C1A["Train SVM from scratch\non X_c1, y_c1"]
            C1B["Serialize via pickle\nReturn bytes · n_c1 · loss"]
            C1A --> C1B
        end

        subgraph C2["Client 2 · 80% DDoS"]
            C2A["Train SVM from scratch\non X_c2, y_c2"]
            C2B["Serialize via pickle\nReturn bytes · n_c2 · loss"]
            C2A --> C2B
        end

        subgraph AGG["FedEnsemble Aggregation"]
            AGG1["client_models = [\n  (SVM_c1, n_c1),\n  (SVM_c2, n_c2)\n]"]
        end

        subgraph INFER["Weighted Soft-Voting Inference"]
            INF1["total = n_c1 + n_c2"]
            INF2["proba = (n_c1/total) · SVM_c1.predict_proba\n     + (n_c2/total) · SVM_c2.predict_proba"]
            INF3["y_pred = argmax(proba, axis=1)"]
            INF1 --> INF2 --> INF3
        end

        EVAL["Server Evaluate\nrecord metrics"]
        BEST{F1 Macro\n> best?}
        CKPT["Save best client_models\nbest_client_models = client_models.copy"]

        C1B & C2B --> AGG --> INFER --> EVAL --> BEST
        BEST -->|Yes| CKPT
        BEST -->|No| AGG
    end

    subgraph OUT["Results"]
        O1[Confusion Matrix PNG]
        O2[svm_server_history.csv]
    end

    MODEL --> ROUND
    CKPT --> OUT
```

---

## 10. All FL Strategies — Side-by-Side Comparison

```mermaid
flowchart LR
    subgraph INPUT["Shared Input\n(All Models)"]
        direction TB
        I1[X_c1, y_c1\nClient 1 Local Data]
        I2[X_c2, y_c2\nClient 2 Local Data]
    end

    subgraph CNN_F["CNN · FedAvg"]
        direction TB
        CN1["train: model.fit\nclass_weight · EarlyStopping\n5 epochs per round"]
        CN2["aggregate: weighted avg\nθ = Σ n_i/N · θ_i\nof all layer weights"]
        CN1 --> CN2
    end

    subgraph DT_F["DT · FedBest"]
        direction TB
        DT1["train: tree.fit\nretrain from scratch\nrandom_state=42"]
        DT2["aggregate: select\nbest client model\nby min training loss"]
        DT1 --> DT2
    end

    subgraph RF_F["RF · FederatedForest"]
        direction TB
        RF1["train: forest.fit\n100 trees per client\nretrain from scratch"]
        RF2["aggregate: merge trees\n100 + 100 = 200 trees\nensemble diversity"]
        RF1 --> RF2
    end

    subgraph LR_F["LR · FedAvg"]
        direction TB
        LR1["train: lr.fit\nwarm-start from\nglobal coef_ + intercept_"]
        LR2["aggregate: weighted avg\ncoef_ = Σ n_i/N · coef_i\nintercept_ = Σ n_i/N · b_i"]
        LR1 --> LR2
    end

    subgraph NB_F["NB · FedAvg"]
        direction TB
        NB1["train: gnb.fit\nrecompute class\nmeans and variances"]
        NB2["aggregate: weighted avg\ntheta_ · var_\nclass_prior_"]
        NB1 --> NB2
    end

    subgraph SVM_F["SVM · FedEnsemble"]
        direction TB
        SV1["train: svc.fit\ntrain from scratch\nprobability=True"]
        SV2["aggregate: keep all\nclient SVMs\nfor soft-voting"]
        SV1 --> SV2
    end

    subgraph EVAL["Server Evaluation\n(all models)"]
        direction TB
        E1["predict on 63,912\nheld-out test samples"]
        E2["compute: accuracy\nprecision · recall\nf1 · recall_normal\nf1_macro"]
        E3["track best round\nby f1_macro\nrestore at end"]
        E1 --> E2 --> E3
    end

    INPUT --> CNN_F & DT_F & RF_F & LR_F & NB_F & SVM_F
    CNN_F & DT_F & RF_F & LR_F & NB_F & SVM_F --> EVAL
```

---

## 11. Full FL Simulation — One Diagram (All Models Together)

```mermaid
flowchart TD
    DS[(CIC-DDoS2019\n426,076 samples)]

    subgraph PREP["Shared Preprocessing Pipeline"]
        direction LR
        P1[Normalize] --> P2["Feature Selection\n5 methods → 65 features"] --> P3["Non-IID Split\n+ SMOTE per client"]
    end

    subgraph CLIENTS["Federated Clients\n2 Docker containers per simulation"]
        direction LR
        CL1["Client 1\n80% Normal\n20% DDoS"]
        CL2["Client 2\n20% Normal\n80% DDoS"]
    end

    subgraph SIMS["6 Independent FL Simulations"]
        direction TB

        subgraph CNN_SIM["CNN · Port 8080 · 10 rounds · FedAvg"]
            CNN_S["Server\nFedAvg · best-round checkpoint"]
            CNN_C1["Client 1\nConv1D + class_weight + EarlyStopping"]
            CNN_C2["Client 2\nConv1D + class_weight + EarlyStopping"]
            CNN_S <-->|"gRPC"| CNN_C1 & CNN_C2
        end

        subgraph DT_SIM["DT · Port 8083 · 5 rounds · FedBest"]
            DT_S["Server\nFedBest · select best-loss client"]
            DT_C1["Client 1\nDecisionTree retrain"]
            DT_C2["Client 2\nDecisionTree retrain"]
            DT_S <-->|"gRPC"| DT_C1 & DT_C2
        end

        subgraph RF_SIM["RF · Port 8084 · 5 rounds · FederatedForest"]
            RF_S["Server\nFederatedForest · merge 200 trees"]
            RF_C1["Client 1\n100 trees"]
            RF_C2["Client 2\n100 trees"]
            RF_S <-->|"gRPC"| RF_C1 & RF_C2
        end

        subgraph LR_SIM["LR · Port 8081 · 10 rounds · FedAvg"]
            LR_S["Server\nFedAvg on coef_ + intercept_"]
            LR_C1["Client 1\nLogistic Regression warm-start"]
            LR_C2["Client 2\nLogistic Regression warm-start"]
            LR_S <-->|"gRPC"| LR_C1 & LR_C2
        end

        subgraph NB_SIM["NB · Port 8082 · 5 rounds · FedAvg"]
            NB_S["Server\nFedAvg on theta_ + var_"]
            NB_C1["Client 1\nGaussianNB refit"]
            NB_C2["Client 2\nGaussianNB refit"]
            NB_S <-->|"gRPC"| NB_C1 & NB_C2
        end

        subgraph SVM_SIM["SVM · Port 8085 · 5 rounds · FedEnsemble"]
            SVM_S["Server\nFedEnsemble · weighted soft-voting"]
            SVM_C1["Client 1\nSVC RBF train"]
            SVM_C2["Client 2\nSVC RBF train"]
            SVM_S <-->|"gRPC"| SVM_C1 & SVM_C2
        end
    end

    subgraph RESULTS["Results per Simulation"]
        direction LR
        R1[Confusion Matrix PNG]
        R2[Per-Round Metrics CSV]
        R3[Round Plot PNG]
        R4[fl_simulation_results.xlsx]
    end

    DS --> PREP --> CLIENTS
    CLIENTS --> CNN_SIM & DT_SIM & RF_SIM & LR_SIM & NB_SIM & SVM_SIM
    CNN_SIM & DT_SIM & RF_SIM & LR_SIM & NB_SIM & SVM_SIM --> RESULTS
```

---

## Rendering Instructions

To use these diagrams in your paper:

**Option 1 — Online renderer (easiest)**
Paste each code block into https://mermaid.live → export as SVG or PNG.

**Option 2 — VS Code**
Install the "Markdown Preview Mermaid Support" extension → preview this file → screenshot or export.

**Option 3 — Command line**

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i fl_architecture_diagrams.md -o diagram.png
```

**For LaTeX paper**: Export each diagram as PDF or high-res PNG, then include with `\includegraphics`.
