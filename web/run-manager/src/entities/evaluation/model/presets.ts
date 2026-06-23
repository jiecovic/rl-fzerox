// web/run-manager/src/entities/evaluation/model/presets.ts
import { allBuiltInCourseIds } from "@/entities/runConfig/ui/sections/tracks/coursePoolModel";
import type {
  ConfigMetadata,
  EvaluationMode,
  EvaluationSourceArtifact,
  ManagedRunConfig,
} from "@/shared/api/contract";
import { managedRunConfigSchema } from "@/shared/api/contract/config";

export type BuiltInEvaluationPresetId = "time_attack_blue_falcon" | "gp_course_blue_falcon";
export type EvaluationPresetId = BuiltInEvaluationPresetId | `custom:${string}`;

export interface EvaluationPreset {
  builtin: boolean;
  cacheKey: string;
  config: ManagedRunConfig;
  id: EvaluationPresetId;
  label: string;
  repeatsPerTarget: number;
  seed: number;
  sourceArtifact: EvaluationSourceArtifact;
  targetMode: EvaluationMode;
}

export interface EvaluationTargetDraft {
  courseIds: string[];
  cupIds: string[];
  difficulties: string[];
  mode: EvaluationMode;
  vehicleIds: string[];
}

export interface EvaluationPresetOverride {
  config?: ManagedRunConfig;
  label?: string;
  repeatsPerTarget?: number;
  seed?: number;
  sourceArtifact?: EvaluationSourceArtifact;
}

export type EvaluationPresetOverrideMap = Partial<
  Record<EvaluationPresetId, EvaluationPresetOverride>
>;

export interface StoredEvaluationPreset {
  config: ManagedRunConfig;
  id: EvaluationPresetId;
  label: string;
  repeatsPerTarget: number;
  seed: number;
  sourceArtifact: EvaluationSourceArtifact;
  targetMode: EvaluationMode;
}

export interface EvaluationPresetStorage {
  customPresets: StoredEvaluationPreset[];
  overrides: EvaluationPresetOverrideMap;
}

const PRESET_OVERRIDE_STORAGE_KEY = "rl-fzerox:evaluation-presets:v1";
const EVALUATION_SEED_MAX = 4_294_967_295;
const BUILT_IN_PRESET_IDS = [
  "time_attack_blue_falcon",
  "gp_course_blue_falcon",
] satisfies BuiltInEvaluationPresetId[];
type GpDifficulty = ManagedRunConfig["tracks"]["gp_difficulties"][number];

export function buildEvaluationPresets({
  customPresets = [],
  defaultConfig,
  metadata,
  overrides = {},
}: {
  customPresets?: readonly StoredEvaluationPreset[];
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
  overrides?: EvaluationPresetOverrideMap;
}): EvaluationPreset[] {
  const allCourses = allBuiltInCourseIds(metadata);
  const blueFalconId = metadata.vehicles.some((vehicle) => vehicle.id === "blue_falcon")
    ? "blue_falcon"
    : (metadata.vehicles[0]?.id ?? defaultConfig.vehicle.selected_vehicle_ids[0]);
  const defaultGpDifficulty = defaultEvaluationGpDifficulty(metadata);
  const defaultGpDifficultyLabel = gpDifficultyLabel(defaultGpDifficulty, metadata);
  const builtIns: EvaluationPreset[] = [
    {
      cacheKey: "time_attack_blue_falcon",
      builtin: true,
      config: {
        ...clonePresetConfig(defaultConfig),
        tracks: {
          ...defaultConfig.tracks,
          baseline_variant_count: 1,
          gp_difficulties: [],
          include_x_cup: false,
          race_mode: "time_attack",
          selected_course_ids: allCourses,
        },
        vehicle: {
          ...defaultConfig.vehicle,
          selected_vehicle_ids: [blueFalconId],
          selection_mode: "fixed",
        },
      },
      id: "time_attack_blue_falcon",
      label: "Time Attack course · Blue Falcon · all courses",
      repeatsPerTarget: 10,
      seed: 2262218583,
      sourceArtifact: "latest",
      targetMode: "time_attack_course",
    },
    {
      cacheKey: "gp_course_blue_falcon",
      builtin: true,
      config: {
        ...clonePresetConfig(defaultConfig),
        tracks: {
          ...defaultConfig.tracks,
          gp_difficulties: [defaultGpDifficulty],
          include_x_cup: false,
          race_mode: "gp_race",
          selected_course_ids: allCourses,
        },
        vehicle: {
          ...defaultConfig.vehicle,
          selected_vehicle_ids: [blueFalconId],
          selection_mode: "fixed",
        },
      },
      id: "gp_course_blue_falcon",
      label: `GP course · ${defaultGpDifficultyLabel} · Blue Falcon · all courses`,
      repeatsPerTarget: 10,
      seed: 2262218583,
      sourceArtifact: "latest",
      targetMode: "gp_course",
    },
  ];
  const normalizedBuiltIns = builtIns.map((preset) =>
    normalizeEvaluationPreset(applyPresetOverride(preset, overrides[preset.id]), metadata),
  );
  const normalizedCustomPresets = customPresets.map((preset) =>
    normalizeEvaluationPreset(
      {
        ...preset,
        builtin: false,
        cacheKey: `${preset.id}:${preset.seed}:${preset.repeatsPerTarget}`,
      },
      metadata,
    ),
  );
  return [...normalizedBuiltIns, ...normalizedCustomPresets];
}

export function clonePresetConfig(config: ManagedRunConfig): ManagedRunConfig {
  return structuredClone(config) as ManagedRunConfig;
}

export function readEvaluationPresetStorage(): EvaluationPresetStorage {
  if (typeof window === "undefined") {
    return emptyPresetStorage();
  }
  const raw = window.localStorage.getItem(PRESET_OVERRIDE_STORAGE_KEY);
  if (raw === null) {
    return emptyPresetStorage();
  }
  try {
    const parsed = JSON.parse(raw);
    return parsePresetStorage(parsed);
  } catch {
    return emptyPresetStorage();
  }
}

export function writeEvaluationPresetStorage(storage: EvaluationPresetStorage) {
  if (typeof window === "undefined") {
    return;
  }
  window.localStorage.setItem(PRESET_OVERRIDE_STORAGE_KEY, JSON.stringify(storage));
}

export function updateEvaluationPresetOverrideMap(
  overrides: EvaluationPresetOverrideMap,
  presetId: EvaluationPresetId,
  patch: EvaluationPresetOverride,
): EvaluationPresetOverrideMap {
  return {
    ...overrides,
    [presetId]: {
      ...(overrides[presetId] ?? {}),
      ...patch,
    },
  };
}

export function storedPresetFromEvaluationPreset(
  preset: EvaluationPreset,
  patch: Partial<Pick<StoredEvaluationPreset, "id" | "label">> = {},
): StoredEvaluationPreset {
  return {
    config: clonePresetConfig(preset.config),
    id: patch.id ?? preset.id,
    label: patch.label ?? preset.label,
    repeatsPerTarget: preset.repeatsPerTarget,
    seed: preset.seed,
    sourceArtifact: preset.sourceArtifact,
    targetMode: preset.targetMode,
  };
}

export function createDefaultEvaluationPreset({
  defaultConfig,
  metadata,
}: {
  defaultConfig: ManagedRunConfig;
  metadata: ConfigMetadata;
}): StoredEvaluationPreset {
  const allCourses = allBuiltInCourseIds(metadata);
  const blueFalconId = metadata.vehicles.some((vehicle) => vehicle.id === "blue_falcon")
    ? "blue_falcon"
    : (metadata.vehicles[0]?.id ?? defaultConfig.vehicle.selected_vehicle_ids[0]);
  const targetMode: EvaluationMode = "time_attack_course";
  return {
    config: normalizeEvaluationPresetConfig(
      {
        ...clonePresetConfig(defaultConfig),
        tracks: {
          ...defaultConfig.tracks,
          baseline_variant_count: 1,
          gp_difficulties: [],
          include_x_cup: false,
          race_mode: "time_attack",
          selected_course_ids: allCourses,
        },
        vehicle: {
          ...defaultConfig.vehicle,
          selected_vehicle_ids: [blueFalconId],
          selection_mode: "fixed",
        },
      },
      metadata,
      targetMode,
    ),
    id: `custom:${crypto.randomUUID()}`,
    label: "Untitled evaluation preset",
    repeatsPerTarget: 10,
    seed: randomEvaluationSeed(),
    sourceArtifact: "latest",
    targetMode,
  };
}

export function customEvaluationPresetFromConfig({
  config,
  label,
  metadata,
  sourceArtifact = "latest",
  targetMode,
}: {
  config: ManagedRunConfig;
  label: string;
  metadata: ConfigMetadata;
  sourceArtifact?: EvaluationSourceArtifact;
  targetMode: EvaluationMode;
}): StoredEvaluationPreset {
  return {
    config: normalizeEvaluationPresetConfig(config, metadata, targetMode),
    id: `custom:${crypto.randomUUID()}`,
    label,
    repeatsPerTarget: 10,
    seed: randomEvaluationSeed(),
    sourceArtifact,
    targetMode,
  };
}

export function randomEvaluationSeed() {
  const values = new Uint32Array(1);
  crypto.getRandomValues(values);
  return values[0] ?? 0;
}

export function evaluationTargetFromConfig(
  config: ManagedRunConfig,
  metadata: ConfigMetadata,
  targetMode: EvaluationMode,
): EvaluationTargetDraft {
  const normalizedConfig = normalizeEvaluationPresetConfig(config, metadata, targetMode);
  const courseIds = normalizedConfig.tracks.selected_course_ids;
  return {
    courseIds: [...courseIds],
    cupIds: cupIdsForCourses(courseIds, metadata),
    difficulties:
      targetMode === "time_attack_course" ? [] : [...normalizedConfig.tracks.gp_difficulties],
    mode: targetMode,
    vehicleIds: [...normalizedConfig.vehicle.selected_vehicle_ids],
  };
}

export function normalizeEvaluationPresetConfig(
  config: ManagedRunConfig,
  metadata: ConfigMetadata,
  targetMode: EvaluationMode,
): ManagedRunConfig {
  const nextConfig = clonePresetConfig(config);
  if (targetMode === "time_attack_course") {
    nextConfig.tracks.race_mode = "time_attack";
    nextConfig.tracks.gp_difficulties = [];
    nextConfig.tracks.include_x_cup = false;
    nextConfig.tracks.baseline_variant_count = 1;
    return nextConfig;
  }
  nextConfig.tracks.race_mode = "gp_race";
  nextConfig.tracks.include_x_cup = false;
  nextConfig.tracks.gp_difficulties = [
    singleEvaluationGpDifficulty(nextConfig.tracks.gp_difficulties, metadata),
  ];
  return nextConfig;
}

function normalizeEvaluationPreset(
  preset: EvaluationPreset,
  metadata: ConfigMetadata,
): EvaluationPreset {
  return {
    ...preset,
    config: normalizeEvaluationPresetConfig(preset.config, metadata, preset.targetMode),
  };
}

function cupIdsForCourses(courseIds: readonly string[], metadata: ConfigMetadata) {
  const selected = new Set(courseIds);
  return metadata.track_cups
    .filter((cup) => cup.course_ids.some((courseId) => selected.has(courseId)))
    .map((cup) => cup.id);
}

function applyPresetOverride(
  preset: EvaluationPreset,
  override: EvaluationPresetOverride | undefined,
): EvaluationPreset {
  if (override === undefined) {
    return preset;
  }
  return {
    ...preset,
    cacheKey: `${preset.cacheKey}:override`,
    config: override.config === undefined ? preset.config : clonePresetConfig(override.config),
    label: override.label ?? preset.label,
    repeatsPerTarget: override.repeatsPerTarget ?? preset.repeatsPerTarget,
    seed: override.seed ?? preset.seed,
    sourceArtifact: override.sourceArtifact ?? preset.sourceArtifact,
    targetMode: preset.targetMode,
  };
}

function defaultEvaluationGpDifficulty(metadata: ConfigMetadata): GpDifficulty {
  const difficulties = metadata.gp_difficulties.map((option) => option.value as GpDifficulty);
  return (
    difficulties.find((difficulty) => difficulty === "master") ?? difficulties.at(-1) ?? "novice"
  );
}

function singleEvaluationGpDifficulty(
  configured: readonly GpDifficulty[],
  metadata: ConfigMetadata,
): GpDifficulty {
  const validDifficulties = new Set(
    metadata.gp_difficulties.map((option) => option.value as GpDifficulty),
  );
  const configuredDifficulty = configured.find((difficulty) => validDifficulties.has(difficulty));
  return configuredDifficulty ?? defaultEvaluationGpDifficulty(metadata);
}

function gpDifficultyLabel(difficulty: GpDifficulty, metadata: ConfigMetadata): string {
  return (
    metadata.gp_difficulties.find((option) => option.value === difficulty)?.label ?? difficulty
  );
}

function emptyPresetStorage(): EvaluationPresetStorage {
  return { customPresets: [], overrides: {} };
}

function parsePresetStorage(value: unknown): EvaluationPresetStorage {
  if (typeof value !== "object" || value === null) {
    return emptyPresetStorage();
  }
  const record = value as Record<string, unknown>;
  return {
    customPresets: parseStoredPresetList(record.customPresets),
    overrides: parsePresetOverrideMap(record.overrides),
  };
}

function parseStoredPresetList(value: unknown): StoredEvaluationPreset[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    const parsed = parseStoredPreset(item);
    return parsed === null ? [] : [parsed];
  });
}

function parseStoredPreset(value: unknown): StoredEvaluationPreset | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const config = managedRunConfigSchema.safeParse(record.config);
  const label = typeof record.label === "string" ? record.label.trim() : "";
  const targetMode = parseEvaluationMode(record.targetMode);
  const sourceArtifact = parseSourceArtifact(record.sourceArtifact);
  const repeatsPerTarget = parsePositiveInteger(record.repeatsPerTarget);
  const seed = parseSeed(record.seed);
  if (
    !config.success ||
    label.length === 0 ||
    targetMode === null ||
    sourceArtifact === null ||
    repeatsPerTarget === null ||
    seed === null
  ) {
    return null;
  }
  return {
    config: config.data,
    id:
      typeof record.id === "string" && record.id.length > 0
        ? (record.id as EvaluationPresetId)
        : (`custom:${label}` as EvaluationPresetId),
    label,
    repeatsPerTarget,
    seed,
    sourceArtifact,
    targetMode,
  };
}

function parsePresetOverrideMap(value: unknown): EvaluationPresetOverrideMap {
  if (typeof value !== "object" || value === null) {
    return {};
  }
  const overrides: EvaluationPresetOverrideMap = {};
  for (const presetId of BUILT_IN_PRESET_IDS) {
    const parsed = parsePresetOverride((value as Record<string, unknown>)[presetId]);
    if (parsed !== null) {
      overrides[presetId] = parsed;
    }
  }
  return overrides;
}

function parsePresetOverride(value: unknown): EvaluationPresetOverride | null {
  if (typeof value !== "object" || value === null) {
    return null;
  }
  const record = value as Record<string, unknown>;
  const override: EvaluationPresetOverride = {};
  const config = managedRunConfigSchema.safeParse(record.config);
  if (config.success) {
    override.config = config.data;
  }
  const label = typeof record.label === "string" ? record.label.trim() : "";
  if (label.length > 0) {
    override.label = label;
  }
  const sourceArtifact = parseSourceArtifact(record.sourceArtifact);
  if (sourceArtifact !== null) {
    override.sourceArtifact = sourceArtifact;
  }
  const repeatsPerTarget = parsePositiveInteger(record.repeatsPerTarget);
  if (repeatsPerTarget !== null) {
    override.repeatsPerTarget = repeatsPerTarget;
  }
  const seed = parseSeed(record.seed);
  if (seed !== null) {
    override.seed = seed;
  }
  return Object.keys(override).length === 0 ? null : override;
}

function parseSourceArtifact(value: unknown): EvaluationSourceArtifact | null {
  return value === "latest" || value === "best" || value === "final" ? value : null;
}

function parseEvaluationMode(value: unknown): EvaluationMode | null {
  return value === "time_attack_course" || value === "gp_course" ? value : null;
}

function parsePositiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

function parseSeed(value: unknown): number | null {
  return typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= EVALUATION_SEED_MAX
    ? value
    : null;
}
