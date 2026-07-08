#!/bin/bash

# 记录提交时间
SUBMIT_TIME=$(date +%s)
SUBMIT_TIME_HUMAN=$(date "+%Y-%m-%d %H:%M:%S")

# 提交 sbatch 任务并获取 job id
JOB_INFO=$(sbatch run_model.sbt)
JOB_ID=$(echo $JOB_INFO | grep -oP '\d+')

# 记录到日志
echo "$JOB_ID,$SUBMIT_TIME,$SUBMIT_TIME_HUMAN" >> job_submit_time.log

echo "Submitted job $JOB_ID at $SUBMIT_TIME_HUMAN"

# 统计所有已完成任务的排队时间和运行时间，输出到 job_times.csv
echo "job编号,排队时间(秒),运行时间(秒),状态变化过程" > job_times.csv
while read line; do
    jid=$(echo $line | cut -d',' -f1)
    submit_ts=$(echo $line | cut -d',' -f2)
    # 获取SLURM的作业信息（所有状态变化）
    states=$(sacct -j "$jid" --format=State --parsable2 --noheader | awk '{print $1}' | paste -sd'→' -)
    info=$(sacct -j "$jid" --format=JobID,Submit,Start,End,Elapsed,State --parsable2 --noheader | grep -E "^$jid\|")
    if [[ -n "$info" ]]; then
        # 解析字段
        start_time=$(echo $info | cut -d'|' -f3)
        elapsed=$(echo $info | cut -d'|' -f5)
        state=$(echo $info | cut -d'|' -f6)
        # 只统计已完成的任务（不含CANCELLED及CANCELLED+）
        if [[ "$state" == "COMPLETED" || "$state" == "FAILED" || "$state" == "TIMEOUT" || "$state" == "OUT_OF_MEMORY" ]]; then
            # 转换start_time为时间戳
            start_ts=$(date -d "$start_time" +%s)
            # 计算排队时间
            queue_time=$((start_ts - submit_ts))
            # 计算运行时间（sacct已给出，格式为HH:MM:SS或D-HH:MM:SS）
            IFS=':-' read -ra tarr <<< "$elapsed"
            if [[ ${#tarr[@]} -eq 4 ]]; then
                runtime=$((10#${tarr[0]}*86400 + 10#${tarr[1]}*3600 + 10#${tarr[2]}*60 + 10#${tarr[3]}))
            elif [[ ${#tarr[@]} -eq 3 ]]; then
                runtime=$((10#${tarr[0]}*3600 + 10#${tarr[1]}*60 + 10#${tarr[2]}))
            else
                runtime=0
            fi
            echo "$jid,$queue_time,$runtime,$states" >> job_times.csv
        fi
    fi
done < job_submit_time.log

echo "所有已完成任务的排队时间、运行时间和状态变化过程已保存到 job_times.csv，可用Excel直接打开。"