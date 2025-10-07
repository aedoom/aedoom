// Copyright 2025 The aedoom Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"
	"sync/atomic"
)

const size = 256 + int(ActionCount)

// PageRank is a pagerank based model
type PageRank struct {
	rng       *rand.Rand
	markov    [][size][size]uint64
	w, h      int
	action    TypeAction
	votes     [ActionCount]uint64
	started   bool
	iteration int
	loop      [size]chan bool
}

// NewPageRank new pagerank model
func NewPageRank() *PageRank {
	p := &PageRank{
		rng: rand.New(rand.NewSource(1)),
	}
	for i := range p.loop {
		p.loop[i] = make(chan bool, 8)
	}
	return p
}

// Process processes a frame
func (p *PageRank) Process(img Frame) (bool, TypeAction) {
	width := img.Frame.Bounds().Max.X
	height := img.Frame.Bounds().Max.Y
	if p.markov == nil {
		w, h := width/8, height/8
		fmt.Println(width, height, w, h, w*h)
		p.markov = make([][size][size]uint64, w*h)
		p.w, p.h = w, h
	}
	index, previous := 0, img.GrayAt(0, 0).Y
	for y := 0; y < height-8; y += 8 {
		for x := 0; x < width-8; x += 8 {
			for yy := 0; yy < 8; yy++ {
				for xx := 0; xx < 8; xx++ {
					current := img.GrayAt(x+xx, y+yy).Y
					p.markov[index][previous][current]++
					p.markov[index][256+p.action][current]++
					for i := range ActionCount {
						p.markov[index][current][256+i]++
					}
					p.markov[index][current][256+p.action]++
					previous = current
				}
			}
			index++
		}
	}

	process := func(loop chan bool, seed int64, index int) {
		rng := rand.New(rand.NewSource(seed))
		current := 0
		var ranks [size]uint64
		for range loop {
			for range 32 * 256 {
				sum := uint64(0)
				for _, value := range p.markov[index][current] {
					sum += value
				}
				if sum == 0 {
					current = rng.Intn(size)
					continue
				}
				total, selected := uint64(0), uint64(rng.Intn(int(sum)))
				for i, value := range p.markov[index][current] {
					total += value
					if selected < total {
						ranks[i]++
						current = i
						break
					}
				}
			}
			r := ranks[256:]
			sum := uint64(0)
			for _, value := range r {
				sum += value
			}
			if sum == 0 {
				continue
			}
			total, selected := uint64(0), uint64(rng.Intn(int(sum)))
			for i, value := range r {
				total += value
				if selected < total {
					atomic.AddUint64(&p.votes[i], 1)
					break
				}
			}
			shift := false
			for i := range ranks {
				if ranks[i] > 256*256 {
					shift = true
					break
				}
			}
			if shift {
				for i := range ranks {
					ranks[i] >>= 1
				}
			}
		}
	}
	if !p.started {
		for i := range size {
			go process(p.loop[i], p.rng.Int63(), i)
		}
		p.started = true
		for i := range p.loop {
			p.loop[i] <- true
		}
	}

	p.iteration++
	sum := uint64(0)
	for _, value := range p.votes {
		sum += value
	}
	if p.iteration > 30 && sum == uint64(size) {
		total, selected := uint64(0), uint64(p.rng.Intn(int(sum)))
		for i, value := range p.votes {
			total += value
			if selected < total {
				p.action = TypeAction(i)
				break
			}
		}
		for i := range p.votes {
			atomic.StoreUint64(&p.votes[i], 0)
		}
		shift := false
	outer:
		for i := range p.markov {
			for ii := range p.markov[i] {
				for iii := range p.markov[i][ii] {
					if p.markov[i][ii][iii] > 256*256 {
						shift = true
						break outer
					}
				}
			}
		}
		if shift {
			for i := range p.markov {
				for ii := range p.markov[i] {
					for iii := range p.markov[i][ii] {
						p.markov[i][ii][iii] >>= 1
					}
				}
			}
		}
		for i := range p.loop {
			p.loop[i] <- true
		}
		p.iteration = 0
		return true, p.action
	}
	return false, ActionCount
}
