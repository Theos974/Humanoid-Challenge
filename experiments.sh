#!/bin/bash
# Every experiment in the README, in order. Each block is independent once its demos exist; training is ~2 h per
# model on an 8 GB laptop GPU, evaluation ~20 min per 48 episodes. The splits/ folder holds the exact held-out
# layouts and clip selections used for the reported numbers.
# usage: bash experiments.sh <block name>      e.g. bash experiments.sh spread
set -eo pipefail
cd "$(dirname "$0")"
P=python
CLIPS="data/clips_s1.json data/clips_s2.json"   # sessions 1 and 2: bowl -> plate (records from process_clips.py)
T=splits/test_layouts.json

train_eval() {  # run dataset seed
  bash scripts/train.sh "$1" "$2" "$3"
  $P scripts/evaluate.py "$1" "$1" --layouts $T
}

case "$1" in
  clips)       # phone clips -> clip records (hand paths, grasp/release, bowl start)
    $P scripts/process_clips.py "$P2R_DATA/session1" s1
    $P scripts/process_clips.py "$P2R_DATA/session2" s2
    $P scripts/process_clips.py "$P2R_DATA/session3_on" s3_on
    $P scripts/process_clips.py "$P2R_DATA/session3_next" s3_next ;;

  zeroshot)    # SmolVLA already fine-tuned on LIBERO, never shown this scene
    $P scripts/evaluate.py zeroshot "$(huggingface-cli download HuggingFaceVLA/smolvla_libero)" --layouts $T ;;

  copy)        # what should the robot copy? same 71 layouts: my hand paths vs a robot-style teacher (2 seeds each)
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_train.json --name human
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_train.json --name scripted --teacher scripted
    $P scripts/build_dataset.py human lists_human.json
    $P scripts/build_dataset.py scripted lists_scripted.json
    for s in 1000 2000; do train_eval human_$s human $s; train_eval scripted_$s scripted $s; done
    $P scripts/compare.py scripted_1000,scripted_2000 human_1000,human_2000 ;;

  quality)     # does cleaner hand data help? smoothing, and the 40 cleanest clips vs 40 random ones
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_train.json --name smooth10 --smooth 10
    $P scripts/build_dataset.py smooth10 lists_smooth10.json && train_eval smooth10 smooth10 1000
    for sel in clean40 random40; do
      $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_$sel.json --name $sel --smooth 10
      $P scripts/build_dataset.py $sel lists_$sel.json && train_eval $sel $sel 1000
    done ;;

  spread)      # how far does one clip go? 10 clips as filmed vs each replayed at 7 nearby bowl positions
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_spread10.json --name spread_x1 --smooth 10
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_spread10.json --name spread_x7 --smooth 10 --spread 7
    $P scripts/build_dataset.py spread_x1 lists_spread_x1.json
    $P scripts/build_dataset.py spread_x7 lists_spread_x7.json
    train_eval spread_x1 spread_x1 1000
    for s in 1000 2000; do train_eval spread_x7_$s spread_x7 $s; done
    $P scripts/compare.py spread_x7_1000,spread_x7_2000 spread_x1 ;;

  coverage)    # success over a grid of bowl positions (needs the 'copy' models and an old 25-clip model)
    for r in human_1000 scripted_1000; do
      $P scripts/evaluate.py grid_$r $r --layouts splits/grid_layouts.json --seeds 2
    done ;;

  replan)      # how many predicted actions to execute before asking the model again
    for k in 1 5 10 25 50; do $P scripts/evaluate.py replan_$k clean40 --layouts $T --n-action-steps $k; done ;;

  probes)      # does the one-sentence model read the instruction? does clutter matter?
    for prompt in "place the bowl on the plate" "move the bowl onto the plate" "put the bowl onto the plate" \
                  "move the bowl away from the plate" "pick up the plate and put it on the bowl" \
                  "pick up the cup and put it on the plate" "open the top drawer of the cabinet"; do
      $P scripts/evaluate.py "prompt_${prompt// /_}" clean40 --layouts $T --prompt "$prompt"
    done
    for d in glazed_rim_porcelain_ramekin akita_black_bowl chocolate_pudding; do
      $P scripts/evaluate.py distract_$d clean40 --layouts $T --distractor $d:-0.05:-0.27
    done ;;

  nudges)      # recovery data: nudged demos replacing vs added to the clean ones, and a same-size clean control
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_clean40.json --name nudge03 --smooth 10 --nudge 0.3
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_clean40.json --name nudge02 --smooth 10 --nudge 0.2
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_clean40.json --name clean_copy --smooth 10 --reset-offset 7
    $P scripts/build_dataset.py nudge03 lists_nudge03.json
    $P scripts/build_dataset.py clean_plus_nudge02 lists_clean40.json lists_nudge02.json
    $P scripts/build_dataset.py clean_plus_copy lists_clean40.json lists_clean_copy.json
    train_eval nudge03 nudge03 1000
    train_eval clean_plus_nudge02 clean_plus_nudge02 1000
    for s in 1000 2000; do train_eval clean_plus_copy_$s clean_plus_copy $s; done ;;

  two)         # two instructions, same scenes: does the instruction choose the behaviour?
    $P scripts/make_demos.py --clips data/clips_s3_on.json --tags splits/tags_two_on.json --name two_on --smooth 10 --spread 2 --min-carry 15 --min-reach 5
    $P scripts/make_demos.py --clips data/clips_s3_next.json --tags splits/tags_two_next.json --name two_next --smooth 10 --spread 2 --goal next --min-carry 15 --min-reach 5
    $P scripts/build_dataset.py two lists_two_on.json lists_two_next.json
    bash scripts/train.sh two two 1000
    $P scripts/evaluate.py two_ask_on two --layouts splits/test_layouts_two.json --seeds 2 --target splits/target_next.json
    $P scripts/evaluate.py two_ask_next two --layouts splits/test_layouts_two.json --seeds 2 --goal next \
      --prompt "pick up the black bowl and place it next to the plate" --target splits/target_next.json ;;

  rawframes)   # policy that sees the real phone frames instead of the simulator
    $P scripts/make_demos.py --clips $CLIPS --tags splits/tags_test.json --name smooth10_test --smooth 10
    $P scripts/raw_frames_dataset.py rawframes lists_smooth10.json
    bash scripts/train.sh rawframes rawframes 1000 8000
    $P scripts/raw_frames_eval.py rawframes_oracle oracle lists_smooth10_test.json --seeds 1
    $P scripts/raw_frames_eval.py rawframes rawframes lists_smooth10_test.json ;;

  figures)
    $P scripts/figures.py figures ;;

  *) echo "blocks: clips zeroshot copy quality spread coverage replan probes nudges two rawframes figures"; exit 1 ;;
esac
