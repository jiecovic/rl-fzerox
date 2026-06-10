// src/rl_fzerox/apps/run_manager/web/src/widgets/configurator/sections/tracks/TrackMinimap.tsx

import { cn } from "@/shared/ui/cn";
import {
  COURSE_MINIMAP_PATHS,
  COURSE_MINIMAP_VIEW_BOX,
} from "@/widgets/configurator/sections/tracks/courseMinimapData";

interface TrackMinimapProps {
  className?: string;
  courseId: string;
  cup: string;
}

export function TrackMinimap({ className, courseId, cup }: TrackMinimapProps) {
  const path = COURSE_MINIMAP_PATHS[courseId];

  return (
    <div className={cn("course-minimap", className)} data-cup={cup}>
      {path === undefined ? (
        <span className="course-minimap-fallback">Preview unavailable</span>
      ) : (
        <svg
          aria-hidden="true"
          preserveAspectRatio="xMidYMid meet"
          viewBox={COURSE_MINIMAP_VIEW_BOX}
        >
          <path className="course-minimap-shadow" d={path} fillRule="evenodd" />
          <path className="course-minimap-track" d={path} fillRule="evenodd" />
        </svg>
      )}
    </div>
  );
}
