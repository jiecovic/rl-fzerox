import {
  CUP_BANNER_LAYERS,
  CUP_BANNER_VIEW_BOX,
} from "@/features/configurator/sections/tracks/cupBannerData";

interface TrackCupBannerProps {
  cupId: string;
  label: string;
  large?: boolean;
}

export function TrackCupBanner({ cupId, label, large = false }: TrackCupBannerProps) {
  const layers = CUP_BANNER_LAYERS[cupId] ?? [];
  return (
    <svg
      aria-hidden="true"
      className={large ? "track-cup-banner-image large" : "track-cup-banner-image"}
      preserveAspectRatio="xMidYMid meet"
      shapeRendering="crispEdges"
      viewBox={CUP_BANNER_VIEW_BOX}
    >
      {layers.map((layer) => (
        <path d={layer.d} fill={layer.fill} fillRule="evenodd" key={`${cupId}-${layer.fill}`} />
      ))}
      <title>{label}</title>
    </svg>
  );
}
