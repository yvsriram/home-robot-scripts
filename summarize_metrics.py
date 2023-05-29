import json
import argparse
import pandas as pd
import os
from enum import IntEnum, auto

class Skill(IntEnum):
    NAV_TO_OBJ = auto()
    GAZE_AT_OBJ = auto()
    PICK = auto()
    NAV_TO_REC = auto()
    GAZE_AT_REC = auto()
    PLACE = auto()

# from src.home_robot.home_robot.agent.ovmm_agent.ovmm_agent import Skill
def format_latex(x):
    if x < 0.1:
        return f'\phantom{{00}}{100*x:.1f}'
    else:
        return f'\phantom{{0}}{100*x:.1f}'

def compute_stats(aggregated_metrics: pd.DataFrame):
    """Compute stage-wise success rate for each task"""
    stats = {}
    try:
        stats['BASELINE.episode_count'] = aggregated_metrics.loc['END.ovmm_place_success']['count']
        stats['BASELINE.episode_success'] = aggregated_metrics.loc['END.ovmm_place_success']['mean']
    except:
        stats['BASELINE.episode_count'] = aggregated_metrics.loc['END.place_success']['count']
        stats['BASELINE.episode_success'] = aggregated_metrics.loc['END.place_success']['mean']
    stats['BASELINE.does_want_terminate'] = aggregated_metrics.loc['END.does_want_terminate']['mean']
    # let's see how many episodes are in each of the stages when the episode ends
    for skill in Skill:
        stats[f'{skill.name}_in_last'] = aggregated_metrics.loc[f'END.is_curr_skill_{skill.name}']['mean']
    # iterate over skills in order to see which ones were attempted, using the last skill as the final skill
    prev_skill = Skill.NAV_TO_OBJ
    stats[f'{Skill(prev_skill).name}_attempted'] = 1.0
    for skill in range(prev_skill + 1, max(Skill) + 1):
        prev_skill_name = Skill(prev_skill).name
        stats[f'{Skill(skill).name}_attempted'] = stats[f'{prev_skill_name}_attempted'] - stats[f'{prev_skill_name}_in_last']
        prev_skill = skill

    # NAV_TO_OBJ success
    if stats['NAV_TO_OBJ_attempted'] - stats['NAV_TO_OBJ_in_last'] > 0:
        nav_to_obj_success_count = aggregated_metrics.loc['NAV_TO_OBJ.ovmm_nav_orient_to_pick_succ']['count'] * aggregated_metrics.loc['NAV_TO_OBJ.ovmm_nav_orient_to_pick_succ']['mean']
        stats['BASELINE.nav_to_obj_success'] = nav_to_obj_success_count / (stats['BASELINE.episode_count'] * stats['NAV_TO_OBJ_attempted'])
        stats['NAV_TO_OBJ.dist_to_goal'] = aggregated_metrics.loc['NAV_TO_OBJ.ovmm_dist_to_pick_goal']['mean']
        stats['NAV_TO_OBJ.rot_dist_to_goal'] = aggregated_metrics.loc['NAV_TO_OBJ.ovmm_rot_dist_to_pick_goal']['mean']
        
    else:
        stats['BASELINE.nav_to_obj_success'] = 0.0

    if stats['NAV_TO_REC_attempted'] - stats['NAV_TO_REC_in_last'] > 0:
        nav_to_rec_success_count = aggregated_metrics.loc['NAV_TO_REC.ovmm_nav_orient_to_place_succ']['count'] * aggregated_metrics.loc['NAV_TO_REC.ovmm_nav_orient_to_place_succ']['mean']
        stats['BASELINE.nav_to_rec_success'] = nav_to_rec_success_count / (stats['BASELINE.episode_count'] * stats['NAV_TO_REC_attempted'])
        stats['NAV_TO_REC.dist_to_goal'] = aggregated_metrics.loc['NAV_TO_REC.ovmm_dist_to_place_goal']['mean']
        stats['NAV_TO_REC.rot_dist_to_goal'] = aggregated_metrics.loc['NAV_TO_REC.ovmm_rot_dist_to_place_goal']['mean']

    else:
        stats['BASELINE.nav_to_rec_success'] = 0.0
    
    if stats['PLACE_attempted'] > 0:
        try:
            place_success_count = aggregated_metrics.loc['END.ovmm_place_success']['count'] * aggregated_metrics.loc['END.ovmm_place_success']['mean']
        except:
            place_success_count = aggregated_metrics.loc['END.place_success']['count'] * aggregated_metrics.loc['END.place_success']['mean']
        stats['BASELINE.place_success'] = place_success_count / (stats['BASELINE.episode_count'] * stats['PLACE_attempted'])
    else:
        stats['BASELINE.place_success'] = 0.0

    stats['latex_str'] = f'{format_latex(stats["BASELINE.nav_to_obj_success"])} & {format_latex(stats["BASELINE.nav_to_rec_success"])} & {format_latex(stats["BASELINE.place_success"])} & {format_latex(stats["BASELINE.episode_success"])} //'
    
    # find indices in dataframe, with stage success in their name and compute success rate
    for k in aggregated_metrics.index:
        if 'stage_success' in k and 'END' in k:
            stats['HAB_LAB.' + k] = aggregated_metrics.loc[k]['mean']
    return stats

def aggregate_metrics(episode_metrics_df: pd.DataFrame):
    """Aggregate metrics for each episode"""
    # drop the columns with string values
    if 'goal_name' in episode_metrics_df.columns:
        episode_metrics_df =episode_metrics_df.drop(columns=['goal_name'])
    
    # compute aggregated metrics for each column, exclude nans to get mean, min, max and count
    aggregated_metrics = episode_metrics_df.agg(['mean', 'min', 'max', 'count'], axis=0)
    return aggregated_metrics.T


def get_metrics_from_jsons(folder_name: str, exp_name:str) -> dict:
    """Read the metrics dict from json"""
    json_filename = os.path.join(folder_name, exp_name, 'episode_results.json')
    if not os.path.exists(json_filename):
        print(f'File {json_filename} does not exist')
        return None
    episode_metrics = json.load(open(json_filename))
    episode_metrics = {e: episode_metrics[e] for e in list(episode_metrics.keys())[:105]}
    episode_metrics_df = pd.DataFrame.from_dict(episode_metrics, orient='index')
    return episode_metrics_df

def get_summary(args: argparse.Namespace):
    results_dfs = {}
    if args.exp_name is not None:
        exp_names = [args.exp_name]
    else:
        exp_names = os.listdir(os.path.join(args.folder_name))    
    for exp_name in exp_names:
        episode_metrics = get_metrics_from_jsons(args.folder_name, exp_name)
        if episode_metrics is not None:
            try:
                aggregated_metrics = aggregate_metrics(episode_metrics)
                stats = compute_stats(aggregated_metrics)

                # add results to dataframe with exp_name as index
                results_dfs[exp_name] = stats
            except Exception as e:
                print(f'Error in {exp_name}: {e}')

    # create dataframe with exp_name as index
    results_df = pd.DataFrame.from_dict(results_dfs, orient='index')
    # sort by column names and row names
    results_df = results_df.sort_index(axis=0).sort_index(axis=1)
    # save results to csv in same folder
    results_df.to_csv(os.path.join(args.folder_name, 'summary.csv'))

    

def main():
    # parse arguments to read folder_name and exp_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_name', type=str, default='data')
    parser.add_argument('--exp_name', type=str, default=None)
    args = parser.parse_args()

    get_summary(args)

if __name__ == '__main__':
    main()