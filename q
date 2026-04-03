    1  ssh
    2  system status ssh
    3  systemd status ssh
    4  systemctl status ssh
    5  sudo apt install ssh
    6  sudo apt install openssh-client 
    7  systemctl enable ssh
    8  systemctl -l --type service --all|grep ssh
    9  sudo apt-get update
   10  sudo apt-get install openssh-server
   11  systemctl enable ssh
   12  systemctl status ssh
   13  ufw
   14  sudo ufw allow shh
   15  h
   16  exit
   17  ssh-copy-id -i ~/.ssh/id_rsa.pub
   18  do-release-upgrade 
   19  sudo apt update
   20  sudo apt upgrade
   21  sudo visudo.
   22  sudo visudo
   23  sudo
   24  sudo apt update
   25  ipa
   26  ipconfig
   27  ifconfig
   28  sudo apt install net-tools
   29  ifconfig
   30  sudo apt update
   31  sudo visudo
   32  sudo upgrade
   33  sudo apt upgrade
   34  sudo apt autoremove
   35  git
   36  sudo apt autoremove
   37  sudo apt install git
   38  ssh-keygen -t ed25519
   39  cat ~/.ssh/id_ed25519.pub 
   40  git clone git@github.com:Shepherd-ITSec/sentinel_ebpf.git
   41  ls
   42  cd sentinel_ebpf/
   43  chmod +x ./scrips/*
   44  chmod +x ./scrips/
   45  chmod +x ./scripts
   46  sudo bash scripts/k3d-setup.sh --install-ebpf
   47  source $HOME/.local/bin/env
   48  echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
   49  scripts/k3d-smoke.sh
   50  sudo bash scripts/k3d-smoke.sh
   51  exit
   52  apt update
   53  sudo apt update
   54  exit
   55  scripts/k3d-smoke.sh
   56  ls
   57  cd sentinel_ebpf/
   58  scripts/k3d-smoke.sh
   59  chmod +x scripts/
   60  scripts/k3d-smoke.sh
   61  bash scripts/k3d-smoke.sh
   62  gitp pull
   63  git pull
   64  bash scripts/k3d-smoke.sh
   65  exit
   66  bash scripts/k3d-smoke.sh
   67  cd sentinel_ebpf/
   68  bash scripts/k3d-smoke.sh
   69  bash scripts/k3d-smoke.sh --build
   70  cd ~
   71  wget https://api2.cursor.sh/updates/download/golden/linux-x64-deb/cursor/2.4
   72  ls
   73  sudo apt install ./2.4
   74  exit
   75  git pull
   76  bash scripts/k3d-smoke.sh --build
   77  bash scripts/k3d-smoke.sh
   78  export GITHUB_TOKEN=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
   79  bash scripts/k3d-smoke.sh
   80  uv lock
   81  curl -LsSf https://astral.sh/uv/install.sh | sh
   82  uv lock
   83  git config --global user.name "Felix Lab"
   84  bash scripts/k3d-smoke.sh
   85  bash scripts/k3d-smoke.sh --build
   86  uv lock
   87  bash scripts/k3d-smoke.sh --build
   88  sudo apt install linux-headers-$(uname -r)
   89  bash scripts/k3d-smoke.sh --build
   90  docker images sentinel-ebpf-probe sentinel-ebpf-detector --format "{{.Repository}}: {{.Size}}"
   91  docker build -f probe/Dockerfile -t sentinel-ebpf-probe:latest .
   92  docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep -E "sentinel-ebpf-probe|sentinel-ebpf-detector"
   93  docker build -f probe/Dockerfile -t sentinel-ebpf-probe:latest .
   94  bash scripts/k3d-smoke.sh
   95  # See probe pods
   96  kubectl get pods -l app.kubernetes.io/component=probe
   97  # Logs from one probe pod
   98  kubectl logs -l app.kubernetes.io/component=probe -c probe --tail=50
   99  bash scripts/k3d-smoke.sh --build
  100  kubectl logs -l app.kubernetes.io/component=probe -c probe --tail=50
  101  kubectl get pods -l app.kubernetes.io/component=probe
  102  helm upgrade --install sentinel-ebpf ./charts/sentinel-ebpf   -n default   --set ui.enabled=true
  103  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  104  cd /home/felix/sentinel_ebpf
  105  docker build -f ui/Dockerfile -t sentinel-ebpf-ui:latest .
  106  k3d image import sentinel-ebpf-ui:latest -c sentinel-ebpf
  107  kubectl -n default rollout restart deployment/sentinel-ebpf-sentinel-ebpf-ui
  108  k3d image import sentinel-ebpf-ui:latest -c sentinel-ebpf
  109  kubectl -n default rollout restart deployment/sentinel-ebpf-sentinel-ebpf-ui
  110  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  111  bash scripts/k3d-smoke.sh --build
  112  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  113  helm upgrade sentinel-ebpf ./charts/sentinel-ebpf -n default   --set ui.enabled=true   --set ui.image.repository=sentinel-ebpf-ui   --set ui.image.tag=latest   --set probe.image.repository=sentinel-ebpf-probe   --set probe.image.tag=latest   --set detector.image.repository=sentinel-ebpf-detector   --set detector.image.tag=latest
  114  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  115  scripts/run-activity-generator.sh default
  116  bash scripts/k3d-setup.sh --ui --build
  117  scripts/run-activity-generator.sh default
  118  docker images
  119  scripts/run-activity-generator.sh default
  120  bash scripts/k3d-smoke.sh --build
  121  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  122  bash scripts/k3d-smoke.sh --build
  123  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  124  bash scripts/k3d-smoke.sh --build
  125  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  126  ./scripts/k3d-setup.sh 
  127  ./scripts/k3d-setup.sh --build --ui
  128  ./scripts/k3d-setup.sh --build --ui --purge
  129  ./scripts/k3d-setup.sh --build --ui
  130  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  131  ./scripts/k3d-setup.sh --build --ui
  132  ./scripts/k3d-setup.sh --ui
  133  ./scripts/k3d-setup.sh --build --ui
  134  . "\home\felix\.cursor-server\bin\linux-x64\3578107fdf149b00059ddad37048220e41681000/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  135  ./scripts/k3d-setup.sh --build --ui
  136  uv run pytest tests/test_speed.py -v
  137  # Port-forward the detector service
  138  kubectl port-forward -n default svc/sentinel-ebpf-sentinel-ebpf-detector 50052:50052
  139  # In another terminal - run tests
  140  uv run pytest tests/test_speed.py -v
  141  uv run pytest tests/test_speed.py -v
  142  bash scripts/run-activity-generator.sh 
  143  ./scripts/k3d-setup.sh --build --ui
  144  . "\home\felix\.cursor-server\bin\linux-x64\f9919bf991f247689f9ead605b5c5a3239a2a790/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  145  ./scripts/k3d-setup.sh --build --ui
  146  uv run python scripts/convert_beth_to_evt1.py test_data/beth/labelled_testing_data.csv   --evt1-out test_data/beth/testing.events.bin   --labels-out test_data/beth/testing.labels.ndjson
  147  ANOMALY_LOG_PATH=test_data/beth/testing.anomalies.jsonl uv run python -m detector.server
  148  uv run python scripts/replay_logs.py test_data/beth/testing.events.bin --target localhost:50051 --pace fast
  149  ./scripts/k3d-setup.sh --build --ui --pruge
  150  ./scripts/k3d-setup.sh --build --ui --purge
  151  . "\home\felix\.cursor-server\bin\linux-x64\511523af765daeb1fa69500ab0df5b6524424610/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  152  uv run python scripts/run_beth_train_test_eval.py   --train-csv test_data/beth/labelled_training_data.csv   --test-csv test_data/beth/labelled_testing_data.csv   --pace fast
  153  uv sync
  154  . "\home\felix\.cursor-server\bin\linux-x64\511523af765daeb1fa69500ab0df5b6524424610/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  155  uv run python scripts/run_beth_train_test_eval.py   --train-csv test_data/beth/labelled_training_data.csv   --test-csv test_data/beth/labelled_testing_data.csv   --pace fast
  156  . "\home\felix\.cursor-server\bin\linux-x64\7d96c2a03bb088ad367615e9da1a3fe20fbbc6a0/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  157  cd /path/to/sentinel_ebpf
  158  chmod +x scripts/run_all_overnight.sh
  159  ./scripts/run_all_overnight.sh
  160  # Optional: tail -f run_all.log
  161  # Disconnect SSH; reattach later and check run_all.log and test_data/beth/run_all_*/
  162  tail -f run_all.lo
  163  tail -f run_all.log
  164  ./scripts/k3d-setup.sh --build --ui --purge
  165  tail -f run_all.log
  166  uv run python scripts/generate_synthetic_evt1_dataset.py   --out-prefix test_data/synthetic/run1   --total-events 200000   --positive-fraction 0.01   --warmup-fraction 0.75   --seed 42
  167  uv sync -extra local -extra dev
  168  uv sync --extra local --extra dev
  169  uv sync --extra local --extra dev --extra detector
  170  ./scripts/run_synthetic_overnight.sh   test_data/synthetic/run1.evt1   test_data/synthetic/run1.labels.ndjson
  171  chmode +x ./scripts/run_synthetic_overnight.sh
  172  chmod +x ./scripts/run_synthetic_overnight.sh
  173  ./scripts/run_synthetic_overnight.sh   test_data/synthetic/run1.evt1   test_data/synthetic/run1.labels.ndjson
  174  tail -f synthetic_run_all.log
  175  ./scripts/run_synthetic_overnight.sh   test_data/synthetic/run1.evt1   test_data/synthetic/run1.labels.ndjson
  176  tail -f synthetic_run_all.log
  177  # stop any old background synthetic run (safe even if none running)
  178  pkill -f "scripts/run_synthetic_eval.py" || true
  179  # start fresh overnight on fixed dataset
  180  ./scripts/run_synthetic_overnight.sh   test_data/synthetic/run2.evt1   test_data/synthetic/run2.labels.ndjson   synthetic_run_all_run2.log
  181  # watch progress (Ctrl+C to stop watching only)
  182  tail -f synthetic_run_all_run2.log
  183  pkill -f "scripts/run_synthetic_eval.py" || true
  184  uv run python scripts/generate_synthetic_evt1_dataset.py   --out-prefix test_data/synthetic/run3   --total-events 200000   --positive-fraction 0.01   --warmup-fraction 0.75   --seed 42
  185  tail -f synthetic_run_all_run2.log
  186  ./scripts/run_synthetic_overnight.sh   test_data/synthetic/run3.evt1   test_data/synthetic/run3.labels.ndjson   synthetic_run_all.log
  187  tail -f synthetic_run_all.log.
  188  tail -f synthetic_run_all.log
  189  pkill -f "scripts/run_synthetic_eval.py" || true
  190  . "\home\felix\.cursor-server\bin\linux-x64\7d96c2a03bb088ad367615e9da1a3fe20fbbc6a0/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  191  ./scripts/run_synthetic_overnight.sh --generate test_data/synthetic/run4
  192  pkill -f "scripts/run_synthetic_eval.py" || true
  193  ./scripts/run_synthetic_overnight.sh --generate test_data/synthetic/run5
  194  [200~cat /proc/cpuinfo ~
  195  cat /proc/cpuinfo 
  196  sudo lshw -C display
  197  kubectl -n sentinel-ebpf get pods -l app.kubernetes.io/component=probe
  198  kubectl -n sentinel-ebpf get pods
  199  kubectl get pods -n default
  200  kubectl get pods -l app.kubernetes.io/component=probe
  201  ./scripts/generate-activity.sh 
  202  kubectl run activity-generator --rm -i --restart=Never   --image=busybox:1.36   --namespace=<your-namespace>   -- sh -c "$(cat scripts/generate-activity.sh)"
  203  kubectl run activity-generator --rm -i --restart=Never   --image=busybox:1.36   -- sh -c "$(cat scripts/generate-activity.sh)"
  204  # Node where the activity pod ran
  205  kubectl get pod -l run=activity-generator -o wide
  206  # Nodes where the probe runs
  207  kubectl get pod -n <sentinel-namespace> -l app.kubernetes.io/component=probe -o wide
  208  kubectl get pod -l run=activity-generator -o wide
  209  kubectl get pod -l app.kubernetes.io/component=probe -o wide
  210  kubectl -n <namespace> logs -l app.kubernetes.io/component=probe -c probe --tail=100
  211  kubectl logs -l app.kubernetes.io/component=probe -c probe --tail=100
  212  ./scripts/k3d-setup.sh --build --ui --purge
  213  curl -fsSL https://downloads.cursor.com/lab/enterprise/cursor-sandbox-apparmor_0.2.0_all.deb -o cursor-sandbox-apparmor.deb
  214  sudo dpkg -i cursor-sandbox-apparmor.deb
  215  sudo apt update
  216  sudo apt upgrade
  217  restart
  218  reboot
  219  kubectl get pods -l app.kubernetes.io/component=detector
  220  kubectl cp sentinel-ebpf-sentinel-ebpf-detector-5dccbd8644-srftq:/var/log/sentinel-ebpf/events.jsonl ./detector-events.jsonl
  221  ./scripts/k3d-setup.sh --build --ui --purge
  222  sudo apt install texlive-latex-base texlive-latex-extra
  223  sudo apt update
  224  sudo apt install texlive-latex-base texlive-latex-extra latexmk
  225  latexmk
  226  ./scripts/k3d-setup.sh --build --ui --purge
  227  kubectl get pods
  228  git lfs install
  229  sudo apt install git-lfs
  230  git lfs install
  231  git lfs trac "*.jsonl"
  232  git lfs track "*.jsonl"
  233  git add .gitattributes
  234  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080 -n default
  235  uv run python scripts/analyze_dataset.py
  236  kubectl cp sentinel-ebpf-sentinel-ebpf-detector-5dccbd8644-srftq:/var/log/sentinel-ebpf/events.jsonl ./events_09_03_26.jsonl
  237  uv run python scripts/analyze_dataset.py
  238  uv run python scripts/feature_attribution.py events.jsonl --event-index 1000 --algorithm kitnet
  239  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --event-index 1000 --algorithm kitnet
  240  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --event-index 10000 --algorithm kitnet
  241  Luv run python scripts/feature_attribution.py events_05_03_26.jsonl --event-index 50000 --algorithm kitnet
  242  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --event-index 50000 --algorithm kitnet
  243  uv run python scripts/feature_attribution.py events_05_03_26.jsonl  --algorithm kitnet
  244  kubctl get pods
  245  kubectl get pods
  246  kubectl cp sentinel-ebpf-sentinel-ebpf-detector-5dccbd8644-p4zmw:/var/log/sentinel-ebpf/events.jsonl ./events_09_03_26.jsonl
  247  uv run python scripts/analyze_dataset.py
  248  uv run python scripts/analyze_dataset.py --diffs 0
  249  uv run python scripts/analyze_dataset.py --h
  250  uv run python scripts/analyze_dataset.py --diffs 0 events_09_03_26.jsonl 
  251  uv run python scripts/compare_replay_scores.py events_09_03_26.jsonl --start-event 2500000
  252  uv run python scripts/run-activity-generator.sh 
  253  scripts/run-activity-generator.sh 
  254  uv run python scripts/feature_attribution.py events.jsonl --help
  255  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event index 5000
  256  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 5000
  257  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 50000
  258  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 367010
  259  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 81870
  260  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 238053
  261  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json aptribtuion.json --event-index 238055
  262  uv run python scripts/feature_attribution.py events_05_03_26.jsonl --json  --event-id 00064c35d4822969-000204a0
  263  python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --checkpoint-at 200000
  264  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --checkpoint-at 200000
  265  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --aatribution-space lograw
  266  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --attribution-space lograw
  267  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --attribution-space scaled
  268  ./scripts/run_commands_overnight.sh -f scripts/commands.txt overnight.log
  269  tail -f overnight.log
  270  ./scripts/run_commands_overnight.sh -f scripts/commands.txt overnight.log
  271  tail -f overnight.log
  272  kill 3255938
  273  tail -f overnight.log
  274  kill 3255938
  275  ps aux | grep run_commands_overnight
  276  kill 3258701
  277  ps aux | grep run_commands_overnight
  278  pgrep -af run_commands_overnight
  279  ps aux | grep '[r]un_commands_overnight'
  280  tail -f overnight.log
  281  pgrep -af 'scripts/compare_replay_scores.py'
  282  kill 3255946
  283  kill 3255943
  284  pgrep -af 'scripts/compare_replay_scores.py'
  285  ./scripts/run_commands_overnight.sh -f scripts/commands.txt overnight.log
  286  cat overnight.log.pid
  287  tail -f overnight.log
  288  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --attribution-space scaled --num-events 10
  289  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm kitnet   --checkpoint ./ckpt_kitnet_200k.pkl   --attribution-space scaled --num-events 100 --json --offset 10
  290  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm halvespacetrees   --checkpoint ./ckpt_kitnet_200k.pkl   --attribution-space scaled  --json 
  291  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm halvespacetrees   --checkpoint-at 200000 --checkpoint ./ckpt_tree_200k.pkl  --attribution-space scaled  --json 
  292  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm halfespacetrees   --checkpoint-at 200000 --checkpoint ./ckpt_tree_200k.pkl  --attribution-space scaled  --json 
  293  uv run python scripts/feature_attribution.py events_05_03_26.jsonl   --event-id 00064c35d4acbf34-000204d5   --algorithm halfspacetrees   --checkpoint-at 200000 --checkpoint ./ckpt_tree_200k.pkl  --attribution-space scaled  --json 
  294  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm zscore   --threshold 0.7   --out-dir test_data/compare_replay_zscore_05
  295  uv run python scripts/analyze_dataset.py   test_data/compare_replay_zscore_05/replay_dump.jsonl   --out test_data/compare_replay_zscore_05/loss_over_samples.png
  296  uv run python scripts/analyze_dataset.py   test_data/compare_replay_zscore_05/replay_dump.jsonl   --out test_data/compare_replay_zscore_05/loss_over_samples.png --help
  297  uv run python scripts/analyze_dataset.py   test_data/compare_replay_zscore_05/replay_dump.jsonl   --out test_data/compare_replay_zscore_05/loss_over_samples.png --diffs 0
  298  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm zscore   --threshold 0.7   --out-dir test_data/compare_replay_zscore_05 --help
  299  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm zscore   --threshold 0.7   --out-dir test_data/compare_replay_zscore_05_scaled --score mode scaled
  300  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm zscore   --threshold 0.7   --out-dir test_data/compare_replay_zscore_05_scaled --score-mode scaled
  301  uv run python scripts/analyze_dataset.py   test_data/compare_replay_zscore_05_scaled/replay_dump.jsonl   --out test_data/compare_replay_zscore_05_scaled/loss_over_samples.png --diffs 0
  302  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm kitnet   --threshold 0.7   --out-dir test_data/compare_replay_kitnet_05 --score-mode raw
  303  tail overnight.log
  304  uv run python scripts/analyze_dataset.py   test_data/compare_replay_kitnet_09/replay_dump.jsonl   --out test_data/compare_replay_kitnet_09/loss_over_samples_clipped.png   --diffs 0   --y-max-percentile 99.9
  305  uv run python scripts/analyze_dataset.py   test_data/compare_replay_kitnet_09/replay_dump.jsonl   --out test_data/compare_replay_kitnet_09/loss_over_samples_clipped.png   --diffs 0   --y-min-percentile 0.1   --y-max-percentile 99.9
  306  uv run python scripts/analyze_dataset.py   test_data/compare_replay_kitnet_09_AE2/replay_dump.jsonl   --out test_data/compare_replay_kitnet_09_AE2/loss_over_samples_clipped.png   --diffs 0   --y-min-percentile 0.1   --y-max-percentile 99.9
  307  ./scripts/run_commands_overnight.sh --help
  308  ./scripts/run_commands_overnight.sh -f ./scripts/commands2.txt 
  309  tail -f commands_overnight.log
  310  ./scripts/run_commands_overnight.sh -f ./scripts/commands2.txt 
  311  tail -f commands_overnight.log
  312  kill $(cat "commands_overnight.log.pid")
  313  uv run python scripts/compare_replay_scores.py   events_05_03_26.jsonl   --algorithm freq1d   --threshold 0.7  --out-dir test_data/compare_replay_freq1d_05_scaled_AE2 --score-mode scaled
  314  shutdown
  315  reboot
  316  sudo reboot
  317  . "\home\felix\.cursor-server\bin\linux-x64\68fbec5aed9da587d1c6a64172792f505bafa250/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  318  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  
  319  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  --max-y 0.99
  320  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  --y-min 0.99
  321  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  --y-max 0.99
  322  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  --y-max 0.99 -y-min 0.1
  323  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_05/replay_dump.jsonl   --out test_data/compare_replay_kitnet_05/loss_over_samples.png --diffs 0  --y-max 0.99 --y-min 0.1
  324  uv run python scripts/analyze_dataset.py  test_data/compare_replay_kitnet_09_scaled/replay_dump.jsonl   --out test_data/compare_replay_kitnet_09_scaled/loss_over_samples2_diff4.png --diffs 4 
  325  ./scripts/run_commands_overnight.sh -f ./scripts/commands2.txt 
  326  tail -f commands_overnight.log
  327  kill $(cat "commands_overnight.log.pid")
  328  tail -f commands_overnight.log
  329  ./scripts/run_commands_overnight.sh -f ./scripts/commands2.txt 
  330  tail -f commands_overnight.log
  331  kill $(head -1 commands_overnight.log.pid); for p in $(tail -n +2 commands_overnight.log.pid); do kill -TERM -$p 2>/dev/null; done
  332  tail -f commands_overnight.log
  333  ./scripts/run_commands_overnight.sh -f ./scripts/commands2.txt 
  334  tail -f commands_overnight.log
  335  uv run python scripts/analyze_dataset.py  test_data/compare_replay_freq1d_09/replay_dump.jsonl --out test_data/compare_replay_freq1d_09/loss_over_samples.png --diffs 0
  336  uv run python scripts/analyze_dataset.py  test_data/compare_replay_freq1d_09_scaled/replay_dump_freq1d.jsonl --out test_data/compare_replay_freq1d_09_scaled/loss_over_samples.png --diffs 0
  337  ./scripts/k3d-setup.sh --help
  338  ./scripts/k3d-setup.sh --purge --ui --build
  339  cat /etc/passwd
  340  strace printf %s "Hello world"
  341  strace cat /etc/passwd
  342  strace -e openat,read,write,close cat /etc/passwd 2>&1 | grep -E "openat|read|write|close"
  343  cat /etc/passwd
  344  ./scripts/k3d-setup.sh --purge --ui --build
  345  df -h
  346  sudo du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
  347  sudo du -h --max-depth=1 /var 2>/dev/null | sort -hr | head -20
  348  sudo du -h --max-depth=1 /var/log 2>/dev/null | sort -hr | head -20
  349  sudo du -h --max-depth=1 /var/lib 2>/dev/null | sort -hr | head -20
  350  sudo du -h --max-depth=1 /var/lib/docker 2>/dev/null | sort -hr | head -20
  351  docker system df
  352  docker system prune -a
  353  docker system df
  354  docker system prune -a
  355  sudo du -sh /var/lib/docker/containers/*/
  356  ls /var/lib/docker
  357  sudo ls /var/lib/docker
  358  sudo du -sh /var/lib/docker/containers/*/
  359  sudo ls /var/lib/docker/containers
  360  sudo du -sh /var/lib/docker/containers/
  361  sudo ls /var/lib/docker/containers/1e79f3e1eb18e9be02a55ba2bbf0192559f31bf54d31fa57ffa5e81af9878603
  362  sudo du -sh /var/lib/docker/containers/ --max-depth=1
  363  sudo du -h /var/lib/docker/containers/ --max-depth=1
  364  sudo sh -c 'truncate -s 0 /var/lib/docker/containers/*/*-json.log'
  365  sudo du -h /var/lib/docker/containers/ --max-depth=1
  366  sudo systemctl restart docker
  367  df -h
  368  helm upgrade sentinel-ebpf ./charts/sentinel-ebpf -n default
  369  kubectl rollout restart daemonset -l app.kubernetes.io/component=probe -n default
  370  cat /etc/hosts
  371  helm upgrade sentinel-ebpf ./charts/sentinel-ebpf -n default
  372  kubectl rollout restart daemonset -l app.kubernetes.io/component=probe -n default
  373  cat /etc/hosts
  374  ls /etc
  375  cat etc/shells
  376  cat /etc/shells
  377  # Port-forward if needed
  378  kubectl port-forward svc/sentinel-ebpf-detector 50052:50052 &
  379  # Fetch full buffer (10000) and search for cat /etc/shells
  380  DETECTOR_EVENTS_URL=http://localhost:50052/recent_events uv run python scripts/check_watch_events.py --limit=10000 comm=cat path=/etc/shells
  381  DETECTOR_POD=$(kubectl -n default get pods -l app.kubernetes.io/component=detector -o jsonpath='{.items[0].metadata.name}')
  382  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep -E 'cat|shells'
  383  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep 'cat' | grep 'shells'
  384  cat /etc/shells
  385  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep 'cat' | grep 'shells'
  386  cat /etc/shells
  387  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep 'cat' | grep 'shells'kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep -E 'cat.*shells|shells.*cat'
  388  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep -E 'cat.*shells|shells.*cat'
  389  helm upgrade sentinel-ebpf ./charts/sentinel-ebpf -n default
  390  kubectl rollout restart daemonset -l app.kubernetes.io/component=probe -n default
  391  cat /etc/shells
  392  kubectl -n default exec "${DETECTOR_POD}" -c detector -- tail -n 50000 /var/log/sentinel-ebpf/events.jsonl | grep -E 'cat.*shells|shells.*cat'
  393  ./scripts/k3d-setup.sh --purge --ui --build
  394  apt update
  395  sudo apt update
  396  sudo apt upgrade
  397  ./scripts/k3d-setup.sh --purge --ui --build
  398  helm upgrade --install sentinel-ebpf ./charts/sentinel-ebpf   --namespace default   --set ui.enabled=true   --set ui.image.repository=sentinel-ebpf-ui   --set ui.image.tag=latest   --set ui.image.pullPolicy=Never
  399  cat /etc/shells
  400  cat /etc/passwd
  401  . "\home\felix\.cursor-server\bin\linux-x64\224838f96445be37e3db643a163a817c15b36060/out/vs/workbench/contrib/terminal/common/scripts/shellIntegration-bash.sh"
  402  source /home/felix/sentinel_ebpf/.venv/bin/activate
  403  cat etc/passwd
  404  cat /etc/passwd
  405  cat /etc/shadow
  406  kubectl port-forward svc/sentinel-ebpf-ui 8080:8080 -n sentinel-ebpf
  407  kubectl port-forward svc/sentinel-ebpf-ui 8080:8080
  408  kubectl get pods
  409  kubectl port-forward svc/sentinel-ebpf-ui-bd797dc98-555qs 8080:8080
  410  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  411  ./scripts/k3d-setup.sh --purge --ui --build
  412  source /home/felix/sentinel_ebpf/.venv/bin/activate
  413  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  414  ./scripts/k3d-setup.sh --purge --ui --build
  415  uv run python -m cProfile -o profile.stats -s cumtime scripts/run_detector_eval.py ...
  416  # or
  417  uv run py-spy top -- python -m detector.server ...
  418  ./scripts/k3d-setup.sh --purge --ui --build
  419  source /home/felix/sentinel_ebpf/.venv/bin/activate
  420  cat /etc/shadow
  421  source /home/felix/sentinel_ebpf/.venv/bin/activate
  422  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  423  helm upgrade sentinel-ebpf ./charts/sentinel-ebpf -f charts/sentinel-ebpf/values.yaml
  424  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  425  kubectl get pods
  426  ./scripts/k3d-setup.sh --purge --ui --build
  427  docker build -f detector/Dockerfile -t sentinel-ebpf-detector:latest .
  428  k3d image import -c sentinel-ebpf sentinel-ebpf-detector:latest
  429  helm upgrade --install sentinel-ebpf ./charts/sentinel-ebpf   --namespace default   --set detector.image.repository=sentinel-ebpf-detector   --set detector.image.tag=latest   --set detector.image.pullPolicy=Never
  430  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  431  kubectl get pods
  432  ./scripts/k3d-setup.sh --purge --ui --build
  433  source /home/felix/sentinel_ebpf/.venv/bin/activate
  434  source /home/felix/sentinel_ebpf/.venv/bin/activate
  435  uv run python scripts/analyze_dataset.py events_17_03_26.jsonl --max-events 200000 --diffs 0
  436  uv run python scripts/feature_attribution.py events_17_03_26.jsonl --event-index 199999
  437  uv run python scripts/feature_attribution.py events_17_03_26.jsonl --algorithm freq1d --attribution-space scaled --event-index 199999
  438  uv run python scripts/feature_attribution.py events_17_03_26.jsonl --algorithm freq1d --attribution-space raw --event-index 199999 --json
  439  tail memstream_ablation_overnight.log
  440  tail -f memstream_ablation_overnight.log
  441  source /home/felix/sentinel_ebpf/.venv/bin/activate
  442  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  443  kubectl -n "${NAMESPACE}" cp "${DETECTOR_POD}:/var/log/sentinel-ebpf/anomalies.jsonl"
  444  kubectl -n "${NAMESPACE}" cp "${DETECTOR_POD}:/var/log/sentinel-ebpf/anomalies.jsonl" "events_17_03_26.jsonl"
  445  kubectl  cp "${DETECTOR_POD}:/var/log/sentinel-ebpf/anomalies.jsonl" "events_17_03_26.jsonl"
  446  kubectl get pods
  447  kubectl  cp "sentinel-ebpf-sentinel-ebpf-detector-77887f5fc7-tqv9v:/var/log/sentinel-ebpf/anomalies.jsonl" "events_17_03_26.jsonl"
  448  kubectl  cp "sentinel-ebpf-sentinel-ebpf-detector-77887f5fc7-tqv9v:/var/log/sentinel-ebpf/evetns.jsonl" "events_17_03_26.jsonl"
  449  kubectl  cp "sentinel-ebpf-sentinel-ebpf-detector-77887f5fc7-tqv9v:/var/log/sentinel-ebpf/events.jsonl" "events_17_03_26.jsonl"
  450  ./scripts/k3d-setup.sh --purge --ui --build
  451  kubectl get pods
  452  ./scripts/k3d-setup.sh --purge --ui --build
  453  source /home/felix/sentinel_ebpf/.venv/bin/activate
  454  ./scripts/k3d-setup.sh --purge --ui --build
  455  ./scripts/run_commands_overnight.sh -f scripts/commands_memstream_loda_17_500k.txt memstream_loda_17_500k.log
  456  tail -f memstream_loda_17_500k.log
  457  source /home/felix/sentinel_ebpf/.venv/bin/activate
  458  ./scripts/k3d-setup.sh --purge --ui --build
  459  uv run python scripts/extract_events_subset.py
  460  uv run python scripts/extract_events_subset.py events_17_03_26.jsonl -o events_17_03_26_1M.jsonl -n 1000000
  461  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M   --score-mode percentile   --limit 100000
  462  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M   --score-mode percentile   --limit 100000 --detector-port 50055
  463  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M   --score-mode percentile   --limit 200000 --detector-port 50055
  464  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M_scaled   --score-mode scaled   --limit 100000 --detector-port 50055
  465  source /home/felix/sentinel_ebpf/.venv/bin/activate
  466  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M   --score-mode percentile   --limit 100000
  467  kubectl get all
  468  kubectl delete all
  469  kubectl delete kubectl delete pods --all
  470  kubectl delete kubectl delete pods all
  471  kubectl delete pods --all
  472  kubectl get all
  473  kubectl delete pods --all
  474  kubectl get all
  475  uv run python scripts/compare_replay_scores.py events_17_03_26_1M.jsonl   --algorithm freq1d   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M   --score-mode percentile   --limit 100000
  476  ./scripts/k3d-setup.sh --purge --ui --build
  477  df -h
  478  uv run python scripts/compare_replay_scores.py events_17_03_26_1M_one_off.jsonl   --algorithm loda_ema   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M_one   --score-mode scaled   --limit 100000
  479  uv run python scripts/compare_replay_scores.py events_17_03_26_1M_one_off.jsonl   --algorithm loda_ema   --threshold 0.7   --out-dir test_data/compare_replay_freq1d_17_1M_one   --score-mode scaled   --limit 100000 --detector-port 50055
  480  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  481  tail -f compare_replay_loda_17.log
  482  kill $(head -1 compare_replay_loda_17.log.pid); for p in $(tail -n +2 compare_replay_loda_17.log.pid); do kill -TERM -$p 2>/dev/null; done
  483  ./scripts/run_commands_overnight.sh -f scripts/commands_compare_replay_loda_17.txt compare_replay_loda_17.log
  484  tail -f compare_replay_loda_17.log
  485  ./scripts/run_commands_overnight.sh -f scripts/commands_compare_replay_loda_17.txt compare_replay_loda_17.log
  486  tail -f compare_replay_loda_17.log
  487  kill $(head -1 compare_replay_loda_17.log.pid); for p in $(tail -n +2 compare_replay_loda_17.log.pid); do kill -TERM -$p 2>/dev/null; done
  488  tail -f compare_replay_loda_17.log
  489  kill $(head -1 compare_replay_loda_17.log.pid); for p in $(tail -n +2 compare_replay_loda_17.log.pid); do kill -TERM -$p 2>/dev/null; done
  490  tail -f compare_replay_loda_17.log
  491  ./scripts/run_commands_overnight.sh -f scripts/commands_compare_replay_loda_17.txt compare_replay_loda_17.log
  492  tail -f compare_replay_loda_17.log
  493  source /home/felix/sentinel_ebpf/.venv/bin/activate
  494  tail -f compare_replay_loda_17.log
  495  ./scripts/run_commands_overnight.sh -f scripts/commands_memstream_percentile_99.txt memstream_percentile_99.log
  496  tail -f memstream_percentile_99.log
  497  ./scripts/k3d-setup.sh --purge --ui --build
  498  source /home/felix/sentinel_ebpf/.venv/bin/activate
  499  kubectl port-forward svc/sentinel-ebpf-sentinel-ebpf-ui 8080:8080
  500  ./scripts/k3d-setup.sh --purge --ui --build
  501  source /home/felix/sentinel_ebpf/.venv/bin/activate
  502  ./scripts/k3d-setup.sh --purge --ui --build
  503  uv run python scripts/replay_lidds.py   --scenario-path /path/to/LID-DS-2021/CVE-2017-7529   --split test   --out-jsonl test_data/lidds/cve-2017-7529.test.jsonl   --target localhost:50051   --pace fast
  504  ls
  505  cd sentinel_ebpf/
  506  ls
  507  cd test
  508  cd test_data/
  509  ls
  510  cd compare_replay_freq1d_09_scaled/
  511  ls
  512  head replay_dump.jsonl 
  513  source /home/felix/sentinel_ebpf/.venv/bin/activate
  514  uv run python scripts/replay_lidds.py   --scenario-path test_data/CVE-2012-2122/CVE-2012-2122   --split test   --out-jsonl test_data/lidds/cve-2012-2122.test.jsonl   --convert-only
  515  ./scripts/run_commands_overnight.sh lidds_replay.log "uv run python scripts/replay_lidds.py --scenario-path test_data/CVE-2012-2122/CVE-2012-2122 --split test --out-jsonl test_data/lidds/cve-2012-2122.test.jsonl --json-only"
  516  ./scripts/run_commands_overnight.sh lidds_replay.log "uv run python scripts/replay_lidds.py --scenario-path test_data/CVE-2012-2122/CVE-2012-2122 --split test --out-jsonl test_data/lidds/cve-2012-2122.test.jsonl --convert-only"
  517  source /home/felix/sentinel_ebpf/.venv/bin/activate
  518  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh artifacts/logs/overnight_convert_lidds.log 'uv run python scripts/replay_lidds.py \
  519    --scenario-path "third_party/LID-DS/scenarios/cve-2012-2122" \
  520    --split test \
  521    --out-jsonl "artifacts/lidds/cve-2012-2122.test.fdattrs.jsonl" \
  522    --convert-only'
  523  Watch: tail -f artifacts/logs/overnight_convert_lidds.log
  524  tail -f artifacts/logs/overnight_convert_lidds.log
  525  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh artifacts/logs/overnight_convert_lidds.log 'uv run python scripts/replay_lidds.py \
  526    --scenario-path "test_data/CVE-2012-2122/CVE-2012-2122" \
  527    --split test \
  528    --out-jsonl "artifacts/lidds/cve-2012-2122.test.fdattrs.jsonl" \
  529    --convert-only'
  530  tail -f artifacts/logs/overnight_convert_lidds.log
  531  source /home/felix/sentinel_ebpf/.venv/bin/activate
  532  tail -f artifacts/logs/overnight_convert_lidds.log
  533  source /home/felix/sentinel_ebpf/.venv/bin/activate
  534  clear
  535  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh artifacts/logs/overnight_train_lidds.log "uv run python /home/felix/sentinel_ebpf/scripts/replay_lidds.py \
  536    --scenario-path /home/felix/sentinel_ebpf/test_data/CVE-2012-2122/CVE-2012-2122 \
  537    --split training \
  538    --rules-path /home/felix/sentinel_ebpf/scripts/replay_lidds_rules.yaml \
  539    --out-jsonl /home/felix/sentinel_ebpf/test_data/lidds/cve-2012-2122.train.jsonl \
  540    --start-detector \
  541    --detector-port 50055 \
  542    --detector-algorithm grimmer_mlp \
  543    --event-dump-path /home/felix/sensinel_ebpf/test_data/lidds/grimmer_mlp_events_dump.jsonl \
  544    --pace fast \
  545    --save-checkpoint /home/felix/sentinel_ebpf/test_data/lidds/grimmer_mlp_training.pkl"
  546  source /home/felix/sentinel_ebpf/.venv/bin/activate
  547  ./scripts/k3d-setup.sh --purge --ui --build
  548  source /home/felix/sentinel_ebpf/.venv/bin/activate
  549  tail -f artifacts/logs/overnight_convert_lidds.log
  550  tail -f artifacts/logs/overnight_train_lidds.log
  551  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh artifacts/logs/overnight_train_lidds.log "uv run python -m scripts.train_detector_checkpoint \
  552    '/home/felix/sentinel_ebpf/test_data/lidds/cve-2012-2122.train.jsonl' \
  553    --algorithm grimmer_mlp \
  554    --out '/home/felix/sentinel_ebpf/test_data/lidds/grimmer_mlp_training.pkl'"
  555  tail -f artifacts/logs/overnight_train_lidds.log
  556  source /home/felix/sentinel_ebpf/.venv/bin/activate
  557  tail -f artifacts/logs/overnight_train_lidds.log
  558  ill -TERM -1939718
  559  kill -TERM -1939718
  560  ill -TERM -1939718
  561  tail -f artifacts/logs/overnight_train_lidds.log
  562  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh -f scripts/commands_sequence_mlp_lidds_31_03.txt artifacts/logs/overnight_31-03_train_sequence_mlp.log
  563  tail -f artifacts/logs/overnight_31-03_train_sequence_mlp.log
  564  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh -f scripts/commands_sequence_mlp_lidds_31_03.txt artifacts/logs/overnight_31-03_train_sequence_mlp.log
  565  tail -f artifacts/logs/overnight_31-03_train_sequence_mlp.log
  566  kill -TERM -2123881
  567  kill $(head -1 artifacts/logs/overnight_31-03_train_sequence_mlp.log.pid); for p in $(tail -n +2 artifacts/logs/overnight_31-03_train_sequence_mlp.log.pid); do kill -TERM -$p 2>/dev/null; done
  568  cd /home/felix/sentinel_ebpf && ./scripts/run_commands_overnight.sh -f scripts/commands_sequence_mlp_lidds_31_03.txt artifacts/logs/overnight_31-03_train_sequence_mlp.log
  569  tail -f artifacts/logs/overnight_31-03_train_sequence_mlp.log
  570  source /home/felix/sentinel_ebpf/.venv/bin/activate
  571  tail -f artifacts/logs/overnight_31-03_train_sequence_mlp.log
  572  source /home/felix/sentinel_ebpf/.venv/bin/activate
  573  tail -f artifacts/logs/overnight_31-03_train_sequence_mlp.log
  574  uv run python -m scripts.score_from_checkpoint   "/home/felix/sentinel_ebpf/test_data/lidds/cve-2012-2122.test.jsonl"   --algorithm sequence_mlp   --checkpoint "/home/felix/sentinel_ebpf/test_data/lidds/sequence_mlp_training.pkl"   --out "/home/felix/sentinel_ebpf/test_data/lidds/sequence_mlp_test_scores_100k.jsonl"   --max-events 100000
  575  kill $(head -1 artifacts/logs/overnight_31-03_train_sequence_mlp.log.pid); for p in $(tail -n +2 artifacts/logs/overnight_31-03_train_sequence_mlp.log.pid); do kill -TERM -$p 2>/dev/null; done
  576  history
  577  history | less
