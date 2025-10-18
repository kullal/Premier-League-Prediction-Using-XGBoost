import React, { useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import { gsap } from "gsap";

const calculateSpotlightValues = radius => ({
  proximity: radius * 0.5,
  fadeDistance: radius * 0.75
});

const updateCardGlowProperties = (card, mouseX, mouseY, glow, radius) => {
  const rect = card.getBoundingClientRect();
  const relativeX = ((mouseX - rect.left) / rect.width) * 100;
  const relativeY = ((mouseY - rect.top) / rect.height) * 100;

  card.style.setProperty('--glow-x', `${relativeX}%`);
  card.style.setProperty('--glow-y', `${relativeY}%`);
  card.style.setProperty('--glow-intensity', glow.toString());
  card.style.setProperty('--glow-radius', `${radius}px`);
};

function GlobalSpotlight({ gridRef, enabled = true, spotlightRadius = 300, glowColor = '34, 104, 57' }) {
  useEffect(() => {
    if (!enabled || !gridRef?.current) return;

    const spotlight = document.createElement('div');
    spotlight.className = 'global-spotlight';
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
      transition: opacity 0.15s linear;
    `;
    document.body.appendChild(spotlight);

    const handleMouseMove = e => {
      const section = gridRef.current;
      if (!section) return;
      const rect = section.getBoundingClientRect();
      const inside =
        e.clientX >= rect.left && e.clientX <= rect.right && e.clientY >= rect.top && e.clientY <= rect.bottom;

      const cards = section.querySelectorAll('.card');

      if (!inside) {
        gsap.to(spotlight, { opacity: 0, duration: 0.25, ease: 'power2.out' });
        cards.forEach(c => c.style.setProperty('--glow-intensity', '0'));
        return;
      }

      const { proximity, fadeDistance } = calculateSpotlightValues(spotlightRadius);
      let minDistance = Infinity;

      cards.forEach(card => {
        const r = card.getBoundingClientRect();
        const cx = r.left + r.width / 2;
        const cy = r.top + r.height / 2;
        const distance = Math.hypot(e.clientX - cx, e.clientY - cy) - Math.max(r.width, r.height) / 2;
        const effectiveDistance = Math.max(0, distance);
        minDistance = Math.min(minDistance, effectiveDistance);

        let glowIntensity = 0;
        if (effectiveDistance <= proximity) glowIntensity = 1;
        else if (effectiveDistance <= fadeDistance)
          glowIntensity = (fadeDistance - effectiveDistance) / (fadeDistance - proximity);

        updateCardGlowProperties(card, e.clientX, e.clientY, glowIntensity, spotlightRadius);
      });

      gsap.to(spotlight, { left: e.clientX, top: e.clientY, duration: 0.08, ease: 'power2.out' });

      const targetOpacity =
        minDistance <= proximity ? 0.6 : minDistance <= fadeDistance ? ((fadeDistance - minDistance) / (fadeDistance - proximity)) * 0.6 : 0;

      gsap.to(spotlight, { opacity: targetOpacity, duration: 0.12, ease: 'power2.out' });
    };

    const handleLeave = () => {
      gridRef.current?.querySelectorAll('.card').forEach(c => c.style.setProperty('--glow-intensity', '0'));
      gsap.to(spotlight, { opacity: 0, duration: 0.25, ease: 'power2.out' });
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseleave', handleLeave);

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseleave', handleLeave);
      spotlight.remove();
    };
  }, [gridRef, enabled, spotlightRadius, glowColor]);

  return null;
}

export default function FeatureList({ features }) {
  const list = features ?? [
    {
      title: "High Accuracy",
      description: "XGBoost-powered predictions with proven performance.",
      label: "Accuracy",
      more: "Try Now!"
    },
    {
      title: "Match Analytics",
      description: "In-depth stats and situational insights for every fixture.",
      label: "Analytics",
      more: "Explore Now!"
    },
    {
      title: "Easy to Use",
      description: "Generate predictions quickly with a simple interface.",
      label: "Usability",
      more: "Get Started!"
    },
  ];

  const gridRef = useRef(null);

  return (
    <div className="py-12">
      <style>
        {`
          .bento-section {
            --glow-x: 50%;
            --glow-y: 50%;
            --glow-intensity: 0;
            --glow-radius: 200px;
            --glow-color: 34, 104, 57;
          }

          .card {
            position: relative;
            transition: box-shadow .25s ease, transform .2s ease;
          }

          .card::after {
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

      <section className="max-w-7xl mx-auto mb-8 px-4">
        <h2 className="text-4xl md:text-5xl font-extrabold text-center mb-4">
          Main Features
        </h2>
        <p className="text-center text-gray-300 max-w-3xl mx-auto mb-8">
          Discover the key features that make our EPL Predictor a powerful tool
          for football enthusiasts and analysts alike.
        </p>

        <div ref={gridRef} className="grid grid-cols-1 md:grid-cols-3 gap-6 items-stretch bento-section">
          <GlobalSpotlight gridRef={gridRef} enabled={true} spotlightRadius={300} glowColor={'34, 104, 57'} />

          {/* Konten kiri */}
          <div className="card bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-3xl p-6 flex flex-col justify-between shadow-2xl overflow-hidden transform transition-transform duration-200 hover:-translate-y-1 cursor-pointer relative z-10"
               style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0 }}>
            <div className="flex items-start gap-4">
              <div>
                <h3 className="text-2xl font-bold">EPL Match Predictor</h3>
                <p className="text-gray-300 mt-2">
                  Accurate, fast, and insightful predictions powered by XGBoost
                  and historical data.
                </p>
              </div>
            </div>

            <div className="mt-6 flex items-end justify-between">
              <img
                src="/assets/Premier-League-Logo-White.png"
                alt="Logo EPL"
                className="w-14 lg:w-24 h-auto opacity-70 hover:opacity-100 transition-opacity duration-300"
              />
              <Link
                to="/predict"
                className="inline-flex items-center gap-2 bg-white/6 hover:bg-white/15 text-white px-4 py-2 rounded-xl border border-white/10 shadow-sm transition"
              >
                Try it now!
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M5 12h14M12 5l7 7-7 7"
                  />
                </svg>
              </Link>
            </div>
          </div>

          {/* Bento cards grid */}
          <div className="col-span-1 md:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-6 items-stretch">
            {list.map((f, idx) => {
              const isFull = idx > 1;
              const articleClass = `relative group bg-gradient-to-br from-white/6 to-transparent border border-white/6 rounded-2xl p-6 hover:shadow-xl transform transition-transform duration-300 hover:-translate-y-1 overflow-hidden flex flex-col cursor-pointer z-0 h-full card ${
                isFull
                  ? "sm:col-span-2 sm:flex-col sm:items-start sm:justify-between"
                  : "sm:col-span-1 sm:flex-col sm:items-start sm:justify-between"
              }`;

              return (
                <Link to="/predict" key={idx} className={articleClass} style={{ '--glow-x': '50%', '--glow-y': '50%', '--glow-intensity': 0 }}>
                  <div className="flex items-start justify-between w-full" style={{ zIndex: 2 }}>
                    <div className="flex items-start gap-4">
                      <div>
                        <h4 className="text-lg font-semibold">{f.title}</h4>
                        <p className="text-sm text-gray-300 mt-2">
                          {f.description}
                        </p>
                      </div>
                    </div>

                    <span className="text-xs font-medium bg-white/6 text-white px-3 py-1 rounded-full border border-white/8">
                      {f.label}
                    </span>
                  </div>

                  <div className="mt-24 flex items-start justify-between w-full" style={{ zIndex: 2 }}>
                    <div className="text-sm text-gray-400">{f.more}</div>
                    <div className="text-xs text-gray-400">→</div>
                  </div>
                </Link>
              );
            })}
          </div>
        </div>
      </section>

      <section className="max-w-7xl mx-auto mb-10 text-center px-4">
        <p className="text-gray-400">
          Note: Features may vary based on the specific implementation and data
          sources used.
        </p>
      </section>
    </div>
  );
}
