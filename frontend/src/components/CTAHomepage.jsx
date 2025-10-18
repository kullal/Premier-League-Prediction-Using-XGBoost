import React, { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";

export default function CTAHomepage() {
  const sectionRef = useRef(null);

  useEffect(() => {
    const grid = sectionRef.current;
    if (!grid) return;

    const spotlightRadius = 260;
    const glowColor = "34, 104, 57"; 

    const spotlight = document.createElement("div");
    spotlight.className = "global-spotlight";
    spotlight.style.cssText = `
      position: fixed;
      width: ${spotlightRadius * 2}px;
      height: ${spotlightRadius * 2}px;
      border-radius: 50%;
      pointer-events: none;
      background: radial-gradient(circle,
        rgba(${glowColor}, 0.14) 0%,
        rgba(${glowColor}, 0.06) 20%,
        transparent 60%);
      z-index: 200;
      opacity: 0;
      transform: translate(-50%, -50%);
      mix-blend-mode: screen;
      transition: opacity 0.12s linear;
    `;
    document.body.appendChild(spotlight);

    const cards = () => grid.querySelectorAll(".card");

    const calc = (mouseX, mouseY, elRect) => {
      const cx = elRect.left + elRect.width / 2;
      const cy = elRect.top + elRect.height / 2;
      const distance = Math.hypot(mouseX - cx, mouseY - cy) - Math.max(elRect.width, elRect.height) / 2;
      return Math.max(0, distance);
    };

    const proximity = spotlightRadius * 0.5;
    const fadeDistance = spotlightRadius * 0.75;

    const onMove = (e) => {
      const rect = grid.getBoundingClientRect();
      const inside =
        e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

      if (!inside) {
        gsap.to(spotlight, { opacity: 0, duration: 0.2, ease: "power2.out" });
        cards().forEach(c => c.style.setProperty("--glow-intensity", "0"));
        return;
      }

      let minDistance = Infinity;
      cards().forEach(card => {
        const r = card.getBoundingClientRect();
        const d = calc(e.clientX, e.clientY, r);
        minDistance = Math.min(minDistance, d);

        const relativeX = ((e.clientX - r.left) / r.width) * 100;
        const relativeY = ((e.clientY - r.top) / r.height) * 100;

        let glow = 0;
        if (d <= proximity) glow = 1;
        else if (d <= fadeDistance) glow = (fadeDistance - d) / (fadeDistance - proximity);

        card.style.setProperty("--glow-x", `${relativeX}%`);
        card.style.setProperty("--glow-y", `${relativeY}%`);
        card.style.setProperty("--glow-intensity", glow.toString());
        card.style.setProperty("--glow-radius", `${spotlightRadius}px`);
      });

      gsap.to(spotlight, { left: e.clientX, top: e.clientY, duration: 0.08, ease: "power2.out" });

      const targetOpacity =
        minDistance <= proximity ? 0.6 : minDistance <= fadeDistance ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.6 : 0;

      gsap.to(spotlight, { opacity: targetOpacity, duration: 0.12, ease: "power2.out" });
    };

    const onLeave = () => {
      cards().forEach(c => c.style.setProperty("--glow-intensity", "0"));
      gsap.to(spotlight, { opacity: 0, duration: 0.18, ease: "power2.out" });
    };

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseleave", onLeave);

    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseleave", onLeave);
      spotlight.remove();
    };
  }, []);

  return (
    <section ref={sectionRef} className="max-w-7xl mx-auto mt-16 mb-8 px-4">
      <style>
        {`
          /* card spotlight variables and pseudo glow */
          .card {
            --glow-x: 50%;
            --glow-y: 50%;
            --glow-intensity: 0;
            --glow-radius: 220px;
            --glow-color: 34, 104, 57;
            position: relative;
            transition: box-shadow .25s ease, transform .2s ease;
          }
          .card::after{
            content: '';
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at var(--glow-x) var(--glow-y),
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.9)) 0%,
              rgba(var(--glow-color), calc(var(--glow-intensity) * 0.4)) 25%,
              transparent 55%);
            border-radius: inherit;
            pointer-events: none;
            z-index: 0;
            mix-blend-mode: screen;
            opacity: calc(var(--glow-intensity));
            transition: opacity .18s linear;
          }
        `}
      </style>

      <h2 className="text-4xl md:text-5xl font-extrabold text-center mb-4">
        Let's Try the Features Now!
      </h2>
      <p className="text-center text-gray-300 max-w-3xl mx-auto mb-8">
        Dive into the features that make our Premier League Predictor a
        game-changer.
      </p>

      <card className="card bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-3xl p-6 flex flex-col md:flex-row items-center md:items-stretch gap-6">
        <div className="flex-1 p-6" style={{ zIndex: 2 }}>
          <h2 className="text-3xl md:text-4xl font-bold">
            Try the Predictor — Fast, Accurate, Insightful
          </h2>
          <p className="text-gray-300 text-sm mt-4 mb-16">
            Real-time predictions powered by XGBoost with clear match analytics
            and situational insights. Simple workflow for fans and analysts.
          </p>
          <Link to={"/predict"}>
            <span className="inline-block py-2 px-8 bg-white/5 backdrop-blur-md border border-white/10 shadow-2xl text-lg font-semibold text-white rounded-full hover:bg-white/15 transition-colors duration-300" style={{ zIndex: 2 }}>
              Get Started!
            </span>
          </Link>
        </div>

        <div className="hidden md:flex w-1/3 p-6 items-center justify-center" style={{ zIndex: 2 }}>
          <div className="w-full max-w-xs">
            <svg viewBox="0 0 140 80" width="100%" height="100%" aria-hidden="true" role="img">
              <defs>
                <linearGradient id="spark" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#34d399" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#10b981" stopOpacity="0.18" />
                </linearGradient>
                <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.16" />
                  <stop offset="100%" stopColor="#059669" stopOpacity="0.02" />
                </linearGradient>
              </defs>

              <rect x="0" y="0" width="140" height="80" rx="10" fill="url(#area)" />

              <g fill="#e2e8f0" fillOpacity="0.06" transform="translate(18,30)">
                <rect x="0" y="18" width="6" height="18" rx="3" />
                <rect x="20" y="10" width="6" height="26" rx="3" />
                <rect x="40" y="6" width="6" height="30" rx="3" />
                <rect x="60" y="12" width="6" height="24" rx="3" />
                <rect x="80" y="16" width="6" height="20" rx="3" />
              </g>

              <path d="M14 54 C 28 44, 44 38, 58 42 C 72 46, 90 52, 126 34"
                fill="none"
                stroke="url(#spark)"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.98"
              />

              <path d="M14 54 C 28 44, 44 38, 58 42 C 72 46, 90 52, 126 34 L126 72 L14 72 Z"
                fill="url(#area)"
              />

              <g fill="#fff" stroke="#34d399" strokeWidth="1">
                <circle cx="14" cy="54" r="2.2" />
                <circle cx="58" cy="42" r="2.2" />
                <circle cx="126" cy="34" r="2.2" />
              </g>
            </svg>
          </div>
        </div>
      </card>
    </section>
  );
}
