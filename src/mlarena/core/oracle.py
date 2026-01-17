import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ActionOracle:
    def __init__(self, config: Dict[str, Any]):
        """
        config: dict like:
          enabled: bool
          model_path: str
          dry_run: bool
          pruning_threshold: float
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        self.model_path = config.get("model_path")
        self.dry_run = config.get("dry_run", False)
        self.threshold = config.get("pruning_threshold", 0.0)
        
        self.predictor = None
        
        if self.enabled:
            if not self.model_path:
                logger.warning("ActionOracle is enabled but no model_path provided.")
            else:
                self._load_model()

    def _load_model(self):
        try:
            # Load AutoGluon predictor (lazy import)
            from autogluon.tabular import TabularPredictor
            self.predictor = TabularPredictor.load(self.model_path)
            logger.info(f"ActionOracle loaded successfully from {self.model_path}")
            
            # Check for binary classification capabilities
            if self.predictor.problem_type != 'binary':
                 logger.warning(f"Oracle model problem type is {self.predictor.problem_type}, expected 'binary'. Predictions might be incorrect.")
            
            # Initial schema load
            self._load_schema()
                
        except Exception as e:
            logger.error(f"Failed to load ActionOracle model from {self.model_path}: {e}")
            self.predictor = None
            self.enabled = False # Auto-disable on failure

    def _load_schema(self):
        """Load or reload feature schema from mcts_oracle.csv headers."""
        try:
            csv_path = Path(self.model_path) / "mcts_oracle.csv"
            if csv_path.exists():
                # logger.debug(f"Loading feature schema from {csv_path}")
                # Read only headers
                all_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
                
                # Exclude non-features
                ignore_cols = {'is_improvement', 'child_score', 'delta_score', 'child_id', 'parent_id'}
                self.feature_schema = [c for c in all_cols if c not in ignore_cols]
            else:
                # Fallback to predictor metadata only if schema not yet set or file disappeared
                if not hasattr(self, 'feature_schema'):
                    logger.warning(f"Schema file {csv_path} not found. Fallback to predictor metadata.")
                    self.feature_schema = self.predictor.feature_metadata_in.get_features()
        except Exception as e:
            logger.warning(f"Failed to load CSV schema: {e}. Fallback to predictor metadata.")
            if not hasattr(self, 'feature_schema') and self.predictor:
                self.feature_schema = self.predictor.feature_metadata_in.get_features()

    def evaluate_actions(self, parent_node: Any, candidate_actions: List[Any]) -> Tuple[List[Any], List[float]]:
        """
        Evaluates a list of candidate actions.
        """
        if not self.enabled or not candidate_actions:
            return candidate_actions, [1.0 / max(1, len(candidate_actions))] * len(candidate_actions)
        
        # LIVE RELOAD: Reload model and schema every time to support hot-swapping
        # This allows training a new Oracle while MCTS is running.
        try:
            from autogluon.tabular import TabularPredictor
            self.predictor = TabularPredictor.load(self.model_path)
            self._load_schema()
        except Exception as e:
            if self.predictor is None:
                logger.error(f"Oracle: Failed to load initial model: {e}")
                return candidate_actions, [1.0 / len(candidate_actions)] * len(candidate_actions)
            else:
                # If we have an old predictor, keep using it instead of failing
                logger.warning(f"Oracle: Failed to reload model (using cached version): {e}")

        # 1. Feature Extraction
        try:
            df = self._prepare_features(parent_node, candidate_actions)
            
            # 2. Schema Alignment (Reindexing)
            if hasattr(self, 'feature_schema') and self.feature_schema:
                # This adds missing columns (as NaN) and removes extras, enforcing exact structure
                df = df.reindex(columns=self.feature_schema)
                
        except Exception as e:
            logger.error(f"Oracle feature preparation failed: {e}")
            return candidate_actions, [1.0 / len(candidate_actions)] * len(candidate_actions)
        
        # 3. Prediction
        try:
            # Get probability of positive class (improvement)
            # AutoGluon predict_proba returns DataFrame with columns [0, 1] usually
            probs_df = self.predictor.predict_proba(df)
            
            # Find positive class column (usually 1 or True)
            pos_label = 1
            if 1 not in probs_df.columns:
                 # Fallback logic for labels
                 pos_label = self.predictor.positive_class
            
            priors = probs_df[pos_label].tolist()
            
        except Exception as e:
            logger.error(f"Oracle prediction failed: {e}")
            return candidate_actions, [1.0 / len(candidate_actions)] * len(candidate_actions)

        # 3. Pruning Logic
        # Combine into pairs and sort by probability descending
        scored_actions = sorted(zip(candidate_actions, priors), key=lambda x: x[1], reverse=True)
        
        accepted_actions = []
        accepted_priors = []
        
        max_actions = self.config.get("max_actions", 0)
        
        if max_actions > 0:
            # STRATEGY: Top-K (Ignore threshold)
            # Take top K actions regardless of their score.
            # This ensures we always explore the best available options, even if the model is pessimistic.
            kept_pairs = scored_actions[:max_actions]
            
            # Logging
            n_pruned = len(scored_actions) - len(kept_pairs)
            best_act, best_prob = kept_pairs[0] if kept_pairs else (None, 0.0)
            best_desc = "None"
            if best_act:
                best_desc = f"{best_act.step_name}:{best_act.variant_name}"
            
            logger.info(f"[ORACLE] Top-{max_actions} Strategy. Evaluated {len(candidate_actions)}. Kept: {len(kept_pairs)}. Best: {best_desc} (P={best_prob:.4f})")
            
            accepted_actions = [x[0] for x in kept_pairs]
            accepted_priors = [x[1] for x in kept_pairs]
            
        else:
            # STRATEGY: Threshold (Cut the tail)
            # Keep everything above threshold.
            for action, prob in scored_actions:
                # Safe description
                act_desc = "Action"
                if hasattr(action, 'step_name'):
                    act_desc = f"{action.step_name}:{action.variant_name}"
                
                should_prune = prob < self.threshold
                
                if self.dry_run:
                    # Log decision but keep everything
                    decision = "PRUNE" if should_prune else "KEEP"
                    logger.info(f"[ORACLE DRY-RUN] Action '{act_desc}' P={prob:.4f} -> Would {decision} (Threshold={self.threshold})")
                    accepted_actions.append(action)
                    accepted_priors.append(prob)
                else:
                    if not should_prune:
                        accepted_actions.append(action)
                        accepted_priors.append(prob)
                    # else: dropped
            
            # Logging for threshold strategy
            n_pruned = len(candidate_actions) - len(accepted_actions)
            # Always log summary to confirm Oracle is active
            logger.info(f"[ORACLE] Threshold Strategy. Evaluated {len(candidate_actions)}. Pruned: {n_pruned} ({n_pruned/len(candidate_actions):.1%}) below threshold {self.threshold}")

        # Safety net: if all pruned, keep everything (or best one)
        # Only relevant for Threshold strategy or if max_actions was somehow 0 but list not empty
        if not accepted_actions and candidate_actions:
             logger.warning("[ORACLE] All actions were pruned! Falling back to original list.")
             return candidate_actions, priors

        return accepted_actions, accepted_priors

    def _prepare_features(self, parent_node: Any, candidate_actions: List[Any]) -> pd.DataFrame:
        rows = []
        
        # Extract context
        # Robust attribute access
        parent_score = getattr(parent_node, 'value_mean', 0.0)
        # If value_mean is 0 (unvisited?), try value_best or just 0
        if parent_score == 0.0 and hasattr(parent_node, 'value_best'):
             parent_score = parent_node.value_best or 0.0
             
        # Assuming MCTSNode structure
        depth = 0
        if hasattr(parent_node, 'state'):
             depth = getattr(parent_node.state, 'depth', 0)
        elif hasattr(parent_node, 'depth'):
             depth = parent_node.depth
             
        prev_action = getattr(parent_node, 'action_from_parent', None)
        prev_action_dict = prev_action.to_record() if prev_action and hasattr(prev_action, 'to_record') else {}
        prev_flat = self._parse_action(prev_action_dict, prefix="prev_")
        
        for action in candidate_actions:
            if hasattr(action, 'to_record'):
                ad = action.to_record()
            else:
                ad = action
                
            row = self._parse_action(ad, prefix="")
            
            row['parent_score'] = parent_score
            row['depth'] = depth
            # prev_duration not easily available in node, using default to match training schema
            row['prev_duration'] = 0.0
            
            row.update(prev_flat)
            rows.append(row)
            
        return pd.DataFrame(rows)

    def _parse_action(self, action_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        if not action_dict: return {}
        flat = {}
        group = action_dict.get("group_name") or action_dict.get("group", "unknown")
        variant = action_dict.get("variant_name") or action_dict.get("variant", "unknown")
        
        flat[f"{prefix}action_group"] = group
        flat[f"{prefix}action_variant"] = variant
        
        config = action_dict.get("config", {})
        for k, v in config.items():
            key = f"{prefix}{group}_{k}"
            if isinstance(v, (list, dict)):
                flat[key] = str(v)
                flat[f"{key}_count"] = len(v)
            else:
                flat[key] = v
        return flat
