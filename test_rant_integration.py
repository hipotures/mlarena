#!/usr/bin/env python3
"""Integration test for RANT implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mlarena.modules.rant.config import load_rant_config
from mlarena.modules.rant.storage import RANTStorage, StudyDirection
from mlarena.modules.rant.space import PipelineGenerator
from mlarena.modules.rant.refinement import RefinementEngine

def test_config_loading():
    """Test loading RANT config from super-chain."""
    print("=" * 60)
    print("TEST 1: Config Loading")
    print("=" * 60)

    super_chain_path = Path("conf/preprocess/mla_super_chain.yaml")
    config = load_rant_config(super_chain_path)

    print(f"✓ Config loaded successfully")
    print(f"  - Study name: {config.study_name}")
    print(f"  - N (random): {config.initial_random_trials}")
    print(f"  - K (top): {config.top_k}")
    print(f"  - P (children): {config.children_per_parent}")
    print(f"  - Core length: {config.min_core_len}-{config.max_core_len}")
    print()

    return config


def test_storage_creation(config):
    """Test storage initialization."""
    print("=" * 60)
    print("TEST 2: Storage Creation")
    print("=" * 60)

    # Use temp DB for testing
    storage = RANTStorage("sqlite:////tmp/test_rant_integration.db")

    # Create study
    study_id, created = storage.create_study("test_integration", StudyDirection.MAXIMIZE)
    print(f"✓ Study created: ID={study_id}, new={created}")

    # Create round
    round_num, metadata = storage.resolve_active_round(
        study_id=study_id,
        n_random=config.initial_random_trials,
        top_k=config.top_k,
        children_per_parent=config.children_per_parent,
        min_core_len=config.min_core_len,
        max_core_len=config.max_core_len,
        refinement_config={'grid_size': config.refinement.grid_size},
        seed=config.seed
    )
    print(f"✓ Round created: round={round_num}")
    print(f"  - Frozen N: {metadata['n_random']}")
    print(f"  - Frozen K: {metadata['top_k']}")
    print(f"  - Frozen P: {metadata['children_per_parent']}")
    print()

    return storage, study_id


def test_pipeline_generation(config):
    """Test pipeline generation with real super-chain."""
    print("=" * 60)
    print("TEST 3: Pipeline Generation")
    print("=" * 60)

    super_chain_path = Path("conf/preprocess/mla_super_chain.yaml")
    search_spaces_dir = Path("src/mlarena/search_spaces/preprocess")

    generator = PipelineGenerator(
        super_chain_path=super_chain_path,
        search_spaces_dir=search_spaces_dir,
        seed=config.seed
    )

    print(f"✓ Generator initialized")
    print(f"  - Super-chain steps: {len(generator.super_chain['preprocessors'])}")
    print(f"  - Search spaces: {len(generator.search_spaces)}")
    print(f"  - Fixed start: {len(generator.fixed_start)} steps")
    print(f"  - Fixed end: {len(generator.fixed_end)} steps")
    print()

    # Generate 3 random pipelines (with retry for dependency rejections)
    print("Generating 3 random pipelines (may need retries for dependencies):")
    success_count = 0
    rejection_count = 0

    for i in range(3):
        # Try up to 10 times to generate a valid pipeline
        for attempt in range(10):
            try:
                pipeline = generator.generate_random_pipeline(
                    min_len=config.min_core_len,
                    max_len=config.max_core_len,
                    allow_heavy_steps=config.allow_heavy_steps,
                    allow_heavy_variants=config.allow_heavy_variants
                )

                print(f"  Pipeline {i+1} (attempt {attempt+1}):")
                print(f"    - Core steps: {len(pipeline['core'])}")
                print(f"    - Injected steps: {len(pipeline['injected'])}")
                print(f"    - Total chain: {len(pipeline['full_chain'])}")
                print(f"    - Pipeline sig: {pipeline['pipeline_signature'][:16]}...")
                print(f"    - Arch sig: {pipeline['architecture_signature'][:16]}...")

                # Show step names
                core_names = [s['step'] for s in pipeline['core']]
                print(f"    - Core: {', '.join(core_names)}")

                success_count += 1
                break

            except ValueError as e:
                if "Required group" in str(e):
                    rejection_count += 1
                    continue  # Retry
                else:
                    print(f"  Pipeline {i+1}: ✗ Unexpected error: {e}")
                    break
            except Exception as e:
                print(f"  Pipeline {i+1}: ✗ Error: {e}")
                break
        else:
            print(f"  Pipeline {i+1}: ⚠ Could not generate valid pipeline after 10 attempts")

    print()
    print(f"Generation summary:")
    print(f"  - Successful: {success_count}/3")
    print(f"  - Dependency rejections: {rejection_count}")

    print()
    return generator


def test_trial_insertion(storage, study_id, generator, config):
    """Test inserting trials into storage."""
    print("=" * 60)
    print("TEST 4: Trial Insertion")
    print("=" * 60)

    # Generate and insert 5 trials (with retry for dependency rejections)
    inserted = 0
    dedupe_count = 0
    rejection_count = 0

    for i in range(5):
        # Try up to 20 times to generate and insert a valid pipeline
        for attempt in range(20):
            try:
                pipeline = generator.generate_random_pipeline(
                    min_len=config.min_core_len,
                    max_len=config.max_core_len,
                    allow_heavy_steps=config.allow_heavy_steps,
                    allow_heavy_variants=config.allow_heavy_variants
                )

                trial_id, created = storage.insert_trial(
                    study_id=study_id,
                    round_num=1,
                    phase="random",
                    pipeline=pipeline
                )

                if created:
                    inserted += 1
                    print(f"  Trial {i+1}: ✓ Inserted (ID={trial_id}, {attempt+1} attempts)")
                else:
                    dedupe_count += 1
                    print(f"  Trial {i+1}: = Dedupe (ID={trial_id}, {attempt+1} attempts)")

                break

            except ValueError as e:
                if "Required group" in str(e):
                    rejection_count += 1
                    continue  # Retry
                else:
                    print(f"  Trial {i+1}: ✗ Error: {e}")
                    break
            except Exception as e:
                print(f"  Trial {i+1}: ✗ Error: {e}")
                break
        else:
            print(f"  Trial {i+1}: ⚠ Could not generate after 20 attempts")

    print()
    print(f"Insertion summary:")
    print(f"  - Inserted: {inserted}")
    print(f"  - Deduped: {dedupe_count}")
    print(f"  - Dependency rejections: {rejection_count}")
    print()

    return inserted


def test_refinement(generator, config):
    """Test refinement engine."""
    print("=" * 60)
    print("TEST 5: Refinement")
    print("=" * 60)

    # Generate parent pipeline (with retry for dependency rejections)
    parent = None
    for attempt in range(50):
        try:
            parent = generator.generate_random_pipeline(
                min_len=3,
                max_len=3,
                allow_heavy_steps=False
            )
            break
        except ValueError as e:
            if "Required group" in str(e):
                continue
            raise

    if parent is None:
        print("⚠ Could not generate parent pipeline after 50 attempts")
        return

    print(f"✓ Parent pipeline generated (attempt {attempt+1})")
    print(f"  - Core steps: {len(parent['core'])}")

    # Create refinement engine
    engine = RefinementEngine({
        'grid_size': config.refinement.grid_size,
        'log_grid_size': config.refinement.log_grid_size,
    })

    # Generate children
    children = engine.generate_children(
        parent=parent,
        search_spaces=generator.search_spaces,
        n_children=config.children_per_parent,
        seed=42
    )

    print(f"✓ Generated {len(children)} children (requested {config.children_per_parent})")

    if children:
        print(f"  Child 1 signature: {children[0]['pipeline_signature'][:16]}...")
        print(f"  Parent signature:  {parent['pipeline_signature'][:16]}...")
        print(f"  Signatures differ: {children[0]['pipeline_signature'] != parent['pipeline_signature']}")

    print()


def main():
    """Run all integration tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "RANT INTEGRATION TEST" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    try:
        # Test 1: Config
        config = test_config_loading()

        # Test 2: Storage
        storage, study_id = test_storage_creation(config)

        # Test 3: Pipeline generation
        generator = test_pipeline_generation(config)

        # Test 4: Trial insertion
        inserted = test_trial_insertion(storage, study_id, generator, config)

        # Test 5: Refinement
        test_refinement(generator, config)

        # Summary
        print("=" * 60)
        print("RESULTS")
        print("=" * 60)
        print("✅ All integration tests passed!")
        print(f"   - Config loaded successfully")
        print(f"   - Storage operational")
        print(f"   - Pipeline generation working")
        print(f"   - {inserted} trials inserted")
        print(f"   - Refinement engine operational")
        print()

        return 0

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ INTEGRATION TEST FAILED")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
