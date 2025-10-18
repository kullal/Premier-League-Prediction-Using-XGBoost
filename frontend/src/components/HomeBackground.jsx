import { Renderer, Program, Mesh, Color, Triangle } from 'ogl';
import { useEffect, useRef } from 'react';

export default function HomeBackground() {
  const base = import.meta.env.BASE_URL || "/";
  return (
    <div className="relative w-screen h-screen">
      <img
        src={`${base}assets/izuddin-helmi-adnan-K5ChxJaheKI-unsplash.jpg`}
        alt=""
        className="absolute inset-0 w-full h-full object-cover opacity-70"
      />
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse at center, rgba(0,0,0,0) 30%, rgba(0,0,0,0.35) 60%, rgba(0,0,0,0.7) 100%)",
          mixBlendMode: "multiply",
        }}
      />
    </div>
  );
}
